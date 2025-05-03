# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file
# file_path = "training_metrics.csv"  # Replace with the actual path to your CSV file
# data = pd.read_csv(file_path)

# # Using subplots to organize all metrics into one figure
# fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

# # Subplot 1: Training and Validation Loss
# axes[0].plot(data['Epoch'], data['Training Loss'], label='Training Loss', marker='o')
# axes[0].plot(data['Epoch'], data['Validation Loss'], label='Validation Loss', marker='s')
# axes[0].set_ylabel('Loss')
# axes[0].set_title('Training vs Validation Loss')
# axes[0].legend()
# axes[0].grid()

# # Subplot 2: Training and Validation Accuracy
# axes[1].plot(data['Epoch'], data['Training accuracy'], label='Training Accuracy', marker='o')
# axes[1].plot(data['Epoch'], data['Validation Accuracy'], label='Validation Accuracy', marker='s')
# axes[1].set_ylabel('Accuracy')
# axes[1].set_title('Training vs Validation Accuracy')
# axes[1].legend()
# axes[1].grid()

# # Subplot 3: F1 Scores and Dice Score
# axes[2].plot(data['Epoch'], data['Training f1 Score'], label='Training F1 Score', marker='o')
# axes[2].plot(data['Epoch'], data['Validation F1 Score'], label='Validation F1 Score', marker='s')
# axes[2].plot(data['Epoch'], data['Dice Score'], label='Dice Score', marker='^')
# axes[2].set_xlabel('Epoch')
# axes[2].set_ylabel('Score')
# axes[2].set_title('F1 Scores and Dice Score')
# axes[2].legend()
# axes[2].grid()

# # Adjust layout
# plt.tight_layout()

# # Save the figure (optional)
# # plt.savefig("analysis_plots.png", dpi=300)

# # Show the plots
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "training_metrics.csv"  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Apply a custom style
plt.style.use('ggplot')  # You can experiment with styles like 'seaborn', 'fivethirtyeight', etc.

# Using subplots to organize all metrics into one figure
fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

# Subplot 1: Training and Validation Loss
axes[0].plot(data['Epoch'], data['Training Loss'], label='Training Loss', marker='o')
axes[0].plot(data['Epoch'], data['Validation Loss'], label='Validation Loss', marker='s')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training vs Validation Loss')
axes[0].legend()
axes[0].grid()

# Annotate minimum validation loss
min_val_loss_epoch = data['Epoch'][data['Validation Loss'].idxmin()]
min_val_loss = data['Validation Loss'].min()
axes[0].annotate(f'Min Loss\nEpoch {min_val_loss_epoch}', 
                 xy=(min_val_loss_epoch, min_val_loss), 
                 xytext=(min_val_loss_epoch+0.5, min_val_loss+10), 
                 arrowprops=dict(facecolor='blue', arrowstyle='->'))

# Subplot 2: Training and Validation Accuracy
axes[1].plot(data['Epoch'], data['Training accuracy'], label='Training Accuracy', marker='o')
axes[1].plot(data['Epoch'], data['Validation Accuracy'], label='Validation Accuracy', marker='s')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training vs Validation Accuracy')
axes[1].legend()
axes[1].grid()

# Annotate maximum validation accuracy
max_val_acc_epoch = data['Epoch'][data['Validation Accuracy'].idxmax()]
max_val_acc = data['Validation Accuracy'].max()
axes[1].annotate(f'Max Accuracy\nEpoch {max_val_acc_epoch}', 
                 xy=(max_val_acc_epoch, max_val_acc), 
                 xytext=(max_val_acc_epoch+0.5, max_val_acc-0.05), 
                 arrowprops=dict(facecolor='green', arrowstyle='->'))

# Subplot 3: F1 Scores and Dice Score
axes[2].plot(data['Epoch'], data['Training f1 Score'], label='Training F1 Score', marker='o')
axes[2].plot(data['Epoch'], data['Validation F1 Score'], label='Validation F1 Score', marker='s')
axes[2].plot(data['Epoch'], data['Dice Score'], label='Dice Score', marker='^')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Score')
axes[2].set_title('F1 Scores and Dice Score')
axes[2].legend()
axes[2].grid()

# Annotate highest Dice Score
max_dice_epoch = data['Epoch'][data['Dice Score'].idxmax()]
max_dice = data['Dice Score'].max()
axes[2].annotate(f'Highest Dice\nEpoch {max_dice_epoch}', 
                 xy=(max_dice_epoch, max_dice), 
                 xytext=(max_dice_epoch+0.5, max_dice-0.05), 
                 arrowprops=dict(facecolor='purple', arrowstyle='->'))

# Adjust layout
plt.tight_layout()

# Save the figure (optional)
# plt.savefig("enhanced_analysis_plots.png", dpi=300)

# Show the plots
plt.show()
