import os
from PIL import Image

# Define the folder containing images
folder_path = 'C:\\Users\\Acer\\Desktop\\Raushan Ranjan 21JE0751\\Crack Detection\\data\\masks'  
rotate_path = 'C:\\Users\\Acer\\Desktop\\Raushan Ranjan 21JE0751\\Crack Detection\\data\\masks\\rotated'
# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image (you can expand the types as needed)
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, filename)
        
        # Open the image
        with Image.open(image_path) as image:
            # Rotate the image by 90 degrees (counterclockwise)
            rotated_image = image.rotate(-90)
            
            # Save the rotated image, you can choose a different folder or rename it
            rotated_image.save(os.path.join(rotate_path, f'rotated_{filename}'))
            
            print(f"Rotated {filename}")

print("All images have been rotated.")
