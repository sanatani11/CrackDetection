from PIL import Image
import os

dir_path = "./Mask"
output_dir = ".\\Mask_resized"
new_size = (800, 600)  


for file_name in os.listdir(dir_path):
    
    if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
        
        with Image.open(os.path.join(dir_path, file_name)) as img:
            img = img.resize(new_size)
            #print(img)
            
            new_file_name = "resized_" + file_name
            #print(os.path.join(output_dir, new_file_name))
            if img.mode == 'P':
                img = img.convert('RGB')
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(os.path.join(output_dir, new_file_name))