import argparse
import csv
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

#import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from accuracy_and_score import calculate_accuracy_and_f1

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        test_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    training_f1_scores = []
    validation_f1_scores = []
    dice_scores = []

    metrics_file = 'training_metrics.csv'
    with open(metrics_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Training accuracy', 'Validation Accuracy', 'Training f1 Score', 'Validation F1 Score', 'Dice Score']) 
    
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    
    n_val = int(len(dataset) * val_percent)
    n_test = int(len(dataset) * test_percent)
    n_train = len(dataset) - n_val - n_test
    # Split into train/val/test
    train_set, temp_set = random_split(dataset, [n_train, n_val + n_test], generator=torch.Generator().manual_seed(0))
    val_set, test_set = random_split(temp_set, [n_val, n_test], generator=torch.Generator().manual_seed(0))
    

    
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

    
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        val_loss_after_epoch = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                print(f"loss: {loss.item()}")
                avg_val_loss, val_score = evaluate(model, val_loader, device, amp)
                val_loss_after_epoch += avg_val_loss
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')

                        
                        scheduler.step(val_score)
        training_accuracy, training_f1score = calculate_accuracy_and_f1(model, train_loader, device)
        model.eval()  # <-- Add this
        with torch.no_grad():  # Also add this to disable gradients
            val_loss, dice_score = evaluate(model, val_loader, device, amp)
            accuracy, f1 = calculate_accuracy_and_f1(model, val_loader, device)
        model.train()
        
        


        training_losses.append(epoch_loss)
        validation_losses.append(val_loss_after_epoch)
        training_accuracies.append(training_accuracy)
        validation_accuracies.append(accuracy)
        training_f1_scores.append(training_f1score)
        validation_f1_scores.append(f1)
        dice_scores.append(dice_score.item())

        with open(metrics_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_loss, val_loss, training_accuracy, accuracy, training_f1score, f1, dice_score.item()])


        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

            print(f"Validation accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")

    model.eval()
    test_loss, test_dice = evaluate(model, test_loader, device, amp)
    test_accuracy, test_f1 = calculate_accuracy_and_f1(model, test_loader, device)

    print('\nTest Results:')
    print(f'Loss: {test_loss:.4f} | Accuracy: {test_accuracy:.2f}%')
    print(f'F1 Score: {test_f1:.4f} | Dice Score: {test_dice:.4f}')

    with open(metrics_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Final Test', None, test_loss, None, test_accuracy, None, test_f1, test_dice.item()])


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--test', '-t', dest='test', type=float, default=10.0,
                        help='Percent of the data that is used as test (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)

    model.to(device=device)
    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        test_percent=args.val / 100,
        amp=args.amp
    )