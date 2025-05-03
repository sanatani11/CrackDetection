import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "training_metrics.csv"  
data = pd.read_csv(file_path)

# Plotting Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(data['Epoch'], data['Training Loss'], label='Training Loss', marker='o')
plt.plot(data['Epoch'], data['Validation Loss'], label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid()


# Plotting Training and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(data['Epoch'], data['Training accuracy'], label='Training Accuracy', marker='o')
plt.plot(data['Epoch'], data['Validation Accuracy'], label='Validation Accuracy', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid()


# Plotting F1 Scores and Dice Score
plt.figure(figsize=(10, 6))
plt.plot(data['Epoch'], data['Training f1 Score'], label='Training F1 Score', marker='o')
plt.plot(data['Epoch'], data['Validation F1 Score'], label='Validation F1 Score', marker='x')
plt.plot(data['Epoch'], data['Dice Score'], label='Dice Score', marker='^')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('F1 Scores and Dice Score')
plt.legend()
plt.grid()
plt.show()
