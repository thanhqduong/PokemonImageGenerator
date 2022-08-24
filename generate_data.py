import csv
import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

def load_images_from_folder():
    folder = 'Images'
    images = []
    blue = []
    green = []
    red = []
    count = 0
    for filename in os.listdir(folder):
        count += 1
        if count % 100 == 0:
          print(count)
        img = cv2.imread(os.path.join(folder,filename), 0)
        
        if img is not None:
            img = cv2.resize(img, (64, 64))
            # plt.imshow(img)
            v = img.reshape(64 * 64)
            v = (v - 127.5)/127.5
            images.append(v)


    b = open('data.csv', 'w')
    a = csv.writer(b)
    a.writerows(images)
    b.close()


load_images_from_folder()