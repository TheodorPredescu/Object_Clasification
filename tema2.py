import cv2
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Suppress the warning
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

# Directory path where dataset is stored
dir = 'NEW_DATASET/'
folders_name = ['AIRPLANE', 'DRONE', 'HELICOPTER', 'UAV']

images_not_used = 0

def extract_features(img, coordinates):

    x2, x1, y2, y1 = map(int, map(float, coordinates))  # Parse coordinates
    image = img[x1:y1, x2:x2+2]  # Crop image based on coordinates
    
    # Resize image to 64x64 pixels
    image = cv2.resize(image, (64, 64))

    # Return flattened image
    return image.flatten()

def load_images(folder_path):

    global images_not_used
    images = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):  # Process only .png files
            img = cv2.imread(os.path.join(folder_path, filename), cv2.COLOR_BGR2RGB)  # Read the image
            coordonates_file = filename.replace('.png', '.txt')  # Find the corresponding coordinates file

            if os.path.exists(os.path.join(folder_path, coordonates_file)):
                with open(os.path.join(folder_path, coordonates_file), 'r') as f:
                    coordinates = f.read().split()  # Read coordinates from the text file
                    image_features = extract_features(img, coordinates)  # Extract features
                    images.append(image_features)
            else:
                print(f"No coordinates file for image: {filename}. Image will not be used.")
                images_not_used += 1

    print(f"Images used: {len(images)}")
    print(f"Images not used: {images_not_used}")
    return np.array(images)

def load_folders_extract_features():

    features = []
    labels = []  # Initialize an empty list for labels
    
    for folder_idx, folder_name in enumerate(folders_name):
        folder_dir = os.path.join(dir, folder_name)  # Get the full path of the folder
        
        if os.path.isdir(folder_dir):
            folder_features = load_images(folder_dir)  # Get the features for the current folder
            features.extend(folder_features)  # Add features to the list
            labels.extend([folder_idx] * len(folder_features))  # Assign the same label to all images in the folder
    
    features = np.array(features)  # Convert features list to numpy array
    labels = np.array(labels)  # Convert labels list to numpy array
    
    print(f"Total images: {len(features)}")
    print(f"Labels shape: {labels.shape}")
    return features, labels


def knn_alg(features, labels):
    
    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, np.array(labels), test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_scaled, y_train)

    # Make predictions and calculate accuracy
    y_pred = knn.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"KNN accuracy: {accuracy * 100:.2f}%")
    return accuracy

def native_bayes_alg(features, labels):
    
    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, np.array(labels), test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train Native Bayes model
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(x_train_scaled, y_train)

    # Make predictions and calculate accuracy
    y_pred = gnb.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Native Bayes accuracy: {accuracy * 100:.2f}%")
    return accuracy

def main():
    # Load features and labels (folders_name corresponds to class labels)
    features, labels = load_folders_extract_features()

    features_knn = features.copy()

    # Train KNN classifier and get accuracy
    accuracy_knn = knn_alg(features_knn, labels)

    accuracy_native_bayes = native_bayes_alg(features, labels)

if __name__ == "__main__":
    main()
