import torch
from unet import UNet  # Replace with your actual U-Net model import
import torch.nn as nn

# Initialize the model
model = model = UNet(n_channels=3, n_classes=2)  # Replace with your model parameters

# Function to count the number of layers
def count_layers(model):
    layer_count = 0
    for layer in model.modules():  # Iterate through all modules in the model
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Upsample)):
            layer_count += 1
    return layer_count

# Get the number of layers
num_layers = count_layers(model)
print(f"Total number of layers: {num_layers}")