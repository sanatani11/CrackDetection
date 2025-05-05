from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np


def calculate_accuracy_and_f1(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to calculate gradients during inference
        for batch in data_loader:
            images, true_masks = batch['image'], batch['mask']

            # Move data to device
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            # Forward pass
            outputs = model(images)

            # Convert outputs to predicted labels
            if model.n_classes == 1:
                preds = torch.sigmoid(outputs.squeeze(1)) > 0.5
            else:
                preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(true_masks.cpu().numpy())

    # Flatten the lists to make calculations
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
    model.train()

    return accuracy, f1
