# Object Classification using KNN and Naive Bayes  

This project is a **machine learning-based classification system** designed to identify objects in airspace, such as **airplanes, drones, helicopters, and UAVs**. It processes image datasets by extracting features and applying **K-Nearest Neighbors (KNN)** and **Naive Bayes** classifiers to categorize the images.  

## 🚀 Features  
- 📸 **Image Processing** – Reads and processes `.png` images and extracts object features.  
- 🎯 **Feature Extraction** – Uses bounding box coordinates to crop and resize images to a standard size.  
- 📊 **Machine Learning Models** – Implements **KNN** and **Naive Bayes** algorithms for classification.  
- 🔄 **Automatic Dataset Handling** – Loads images from labeled folders and assigns class labels dynamically.  
- 📈 **Performance Evaluation** – Computes accuracy to assess model performance.  

## 📂 Dataset Structure  
The dataset is stored in a directory (`NEW_DATASET/`) containing subfolders:  

Each image has a corresponding `.txt` file containing bounding box coordinates.

## 🏗️ How It Works  
1. **Image Loading** – Reads images and their respective coordinate files.  
2. **Feature Extraction** – Crops the image based on coordinates and resizes it.  
3. **Model Training** – Trains classifiers (KNN and Naive Bayes) on extracted features.  
4. **Evaluation** – Splits data into training/testing sets and calculates accuracy.  

## 🛠️ Installation & Usage  
```bash
# Clone the repository
git clone https://github.com/TheodorPredescu/tema2_TIA
cd tema2_TIA

# Install dependencies
pip install numpy opencv-python scikit-learn

# Run the classification script
python script.py
