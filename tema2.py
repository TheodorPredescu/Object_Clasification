import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


dir = 'images/'
folders_name = []

def loadImages(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(folders_name.index(folder_path[len(dir):]))

def loadFolders(): 
    for folder_name in folders_name:
        folder_dir = dir + folder_name
        if os.path.isdir(folder_dir):
            loadImages(folder_dir)
        print("Loading images...")
