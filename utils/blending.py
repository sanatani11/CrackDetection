import logging
import cv2
import numpy as np
import os

# img1 = cv2.imread('..\\asset\\0.jpg')
# img2 = cv2.imread('..\\asset\\0_OUT.jpg')
# dst = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

# cv2.imwrite("..\\asset\\blended_image.jpg",dst)
# cv2.imshow('Blended_image',dst)


def blending(img1, img2, filename):
    dst = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    
    imagename = filename.split('.')[0] 
    print(imagename)
    imagename = filename.split('/')[-1]
    print(imagename)
    
    
    
    # Save blended image
    if not cv2.imwrite(f'.\\assets\\Blended_Output\\blended_{imagename}.jpg', dst):
        logging.error(f'Failed to save blended image at ')
    else:
        logging.info(f'Blended image is saved at ')


    
    return dst