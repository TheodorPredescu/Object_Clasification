import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score


dir = 'images/'
folders_name = []

def loadImages(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

def loadFolders(): 
    for folder_name in folders_name:
        folder_dir = dir + folder_name
        if os.path.isdir(folder_dir):
            loadImages(folder_dir)
        print("loading images...")
print("Loading images...")
