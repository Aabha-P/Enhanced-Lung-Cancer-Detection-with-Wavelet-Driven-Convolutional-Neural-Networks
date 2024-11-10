# Enhanced-Lung-Cancer-Detection-with-Wavelet-Driven-Convolutional-Neural-Networks
This repository implements a hybrid CNN model for lung cancer classification using CT scan images. It combines CNNs with Gaussian filters and wavelet transformations to classify images into Normal, Benign, and Malignant categories. The model achieves 99% accuracy and uses data augmentation and advanced evaluation metrics.
# Project Structure
```
└── 📁Wavelet-Powered-CNN-for-Lung-Cancer-Classification
    └── 📁Benign cases
    └── 📁data processed
        └── 📁Excel_files_for_training_and_testing
        └── 📁final filter
            └── 📁Benign
            └── 📁Malignant
            ├── 📁Normal
        └── 📁filters
            └── 📁Benign                
            └── 📁Malignant
            └── 📁Normal
        └── 📁Pywavelet
            └── 📁Benign
            └── 📁Malignant
            └── 📁Normal
        └── 📁SVD
            └── 📁Bengin
            └── 📁Malignant
            ├── 📁Normal
    └── 📁Malignant cases
    └── 📁Normal cases
    └── datapreprocessing.ipynb
    └── dimension.txt
    └── image_data.xlsx
    └── IQ-OTH_NCCD lung cancer dataset.txt
    └── model train test.ipynb
    └── .gitignore
    └── info.txt
    └── README.md
```

# Project Overview

This project leverages image data from lung CT scans to classify three categories of lung cancer :

Benign
Malignant
Normal
The classification is carried out using a Convolutional Neural Network (CNN) with added pre-processing steps like Gaussian filtering and Wavelet Transforms to enhance image features. The final trained model outputs key metrics like accuracy, precision, recall, and ROC-AUC, and provides a confusion matrix for evaluation.

CHECK info.txt IN THE ROOT FOLDER FOR THE DATASET
# Setup Instructions
# # Requirements
The following Python packages are required to run the project:

pandas
numpy
pillow
opencv-python
pywavelets
matplotlib
scipy



Dataset
Place the lung CT scan images in the ROOT folder, organized into subfolders as follows:

Benign/: Contains images classified as benign

Malignant/: Contains images classified as malignant

Normal/: Contains images classified as normal

Running the Project


Preprocessing: Run the scripts in the data_preprocessing/ folder sequentially to load, augment, and apply wavelet transformations and filters to the dataset:

Model Training: After preprocessing, run the script in the model/ folder to define and train the CNN model:

model train test.ipynb: Defines the CNN architecture and trains the model on the processed dataset. It also generates performance metrics and stores results in an Excel file.



# Model Summary
This lung cancer classification model follows these steps:

1. Data Collection and Labeling: The dataset includes CT scan images, labeled into three categories: Normal, Benign, and Malignant.

2. Image Preprocessing:
   - Images are resized to 512x512 using Singular Value Decomposition (SVD) to standardize input size.
   - Wavelet transformations, including bior1.1, haar, and db1, are applied for feature extraction. Images are also augmented into horizontal, vertical, and diagonal sections.
   - Gaussian and Sobel filters are used to reduce noise and enhance important features.

3. Augmentation: Image augmentation is applied to expand the dataset and prevent overfitting.

4. Data Storage: Processed images are stored in separate folders for each category (Normal, Benign, Malignant) for model training.

This pipeline prepares the data for training a model that can accurately classify lung cancer images into Normal, Benign, or Malignant categories.

Evaluation metrics for the model:

Accuracy: Proportion of correct predictions.
Precision: Correct positive predictions / total predicted positives.
Recall: Correct positive predictions / actual positives.
F1-Score: Harmonic mean of precision and recall.
Specificity: Correct negative predictions / total actual negatives.
ROC-AUC: Measures the model’s ability to distinguish between classes.
Confusion Matrix: Shows actual vs predicted classifications.
