import os
import shutil
import pandas as pd
import matplotlib
import cv2
from PIL import Image

veri1=open("D:\\Mocococo\B-2\\Gender\\5-folds\\male_namesOriginal.txt","r")
veri0=open("D:\\Mocococo\B-2\\Gender\\5-folds\\female_namesOriginal.txt","r")

satirlar1=veri1.readlines()
satirlar0=veri0.readlines()

for i in range(len(satirlar1)):   
     path=f"D:/Mocococo/B-2/original_images/{satirlar1[i]}.jpg"
     path = path.replace("\n", "")
     
     im=Image.open(path)
     new_resize_im=im.resize((512, 512))
     
     a = satirlar1[i].replace("\n", "")
     matplotlib.image.imsave(f"D:/Mocococo/B-2/Resize_f-m_images/1/{a}.jpg", new_resize_im)
veri1.close()
