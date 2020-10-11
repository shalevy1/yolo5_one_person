import cv2
import numpy as np

def signal_visualization(is_valid):
    img_red = np.ones((300,300,3),np.uint8)
    img_green = np.copy(img_red)
    img_green[:,:]= [50,255,0]
    img_red[:,:] = [50,0,255]
    img = img_red
    if is_valid:
        img = img_green
    cv2.imshow('image',img)
    pass
