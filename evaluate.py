import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss


# @torch.inference_mode()
# def evaluate(net, dataloader, device, amp):
#     net.eval()
#     num_val_batches = len(dataloader)
#     val_loss = 0
#     dice_score = 0
#     criterion = torch.nn.CrossEntropyLoss() if net.n_classes > 1 else torch.nn.BCEWithLogitsLoss()

#     with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
#         for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#             image, mask_true = batch['image'], batch['mask']

#             image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
#             mask_true = mask_true.to(device=device, dtype=torch.long)

#             mask_pred = net(image)

#             if net.n_classes == 1:
#                 assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
#                 val_loss += criterion(mask_pred.squeeze(1), mask_true.float()).item()
#                 mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
#                 dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
#             else:
#                 assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
#                 val_loss += criterion(mask_pred, mask_true).item()
#                 mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
#                 mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
#                 dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

#     net.train()
#     avg_dice_score = dice_score / max(num_val_batches, 1)
#     avg_val_loss = val_loss / max(num_val_batches, 1)
#     return avg_val_loss, avg_dice_score

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, dice_loss_fn=None):
    net.eval()
    num_val_batches = len(dataloader)
    val_loss = 0
    dice_score = 0
    criterion = torch.nn.CrossEntropyLoss() if net.n_classes > 1 else torch.nn.BCEWithLogitsLoss()

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                bce_loss = criterion(mask_pred.squeeze(1), mask_true.float())
                dice_loss = dice_loss_fn(F.sigmoid(mask_pred.squeeze(1)), mask_true.float(), multiclass=False)
                val_loss += bce_loss.item() + dice_loss.item()
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                bce_loss = criterion(mask_pred, mask_true)
                dice_loss = dice_loss_fn(F.softmax(mask_pred, dim=1).float(),
                                         F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                                         multiclass=True)
                val_loss += bce_loss.item() + dice_loss.item()
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    avg_dice_score = dice_score / max(num_val_batches, 1)
    avg_val_loss = val_loss / max(num_val_batches, 1)
    return avg_val_loss, avg_dice_score
