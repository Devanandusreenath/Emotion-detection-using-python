# Emotion Detection System

## Overview
This project implements an **emotion detection system** using a **Convolutional Neural Network (CNN)** trained on facial expressions. It detects emotions from real-time webcam input and classifies them into one of seven categories.

## Features
- **Face Detection**: Uses OpenCV's Haar Cascade classifier.
- **Emotion Classification**: CNN model trained on grayscale 48x48 images.
- **Real-Time Detection**: Processes video feed from a webcam.
- **Multi-Class Prediction**: Identifies emotions including:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise

## Technologies Used
- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Data Handling**: NumPy, Pandas
- **Model Architecture**: CNN with multiple convolutional layers

## Dataset
- The dataset consists of grayscale images (48x48 pixels) categorized into seven emotion classes.
- The dataset is structured as follows:
  ```
  images/
  ├── train/
  │   ├── angry/
  │   ├── disgust/
  │   ├── fear/
  │   ├── happy/
  │   ├── neutral/
  │   ├── sad/
  │   ├── surprise/
  ├── validation/
  │   ├── angry/
  │   ├── disgust/
  │   ├── fear/
  │   ├── happy/
  │   ├── neutral/
  │   ├── sad/
  │   ├── surprise/
  ```

## Installation
### Prerequisites
Ensure Python (3.7 or later) is installed, along with the required dependencies.

### Steps to Set Up
1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-repo/emotion-detector.git
   cd emotion-detector
   ```
2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Train the model**:
   ```sh
   python train.py
   ```
   - The trained model is saved as `emotiondetector.h5`.
4. **Run real-time emotion detection**:
   ```sh
   python detect.py
   ```

## Usage
### Training the Model
- The `train.py` script loads the dataset, preprocesses images, and trains a CNN model.
- The trained model is saved in JSON (`emotiondetector.json`) and HDF5 (`emotiondetector.h5`) formats.

### Running the Emotion Detector
- The `detect.py` script loads the trained model and processes live webcam input.
- The detected emotion is displayed on the video feed.

## Model Architecture
- **Input**: 48x48 grayscale image.
- **Convolutional Layers**: Extract features from images.
- **Pooling Layers**: Reduce dimensionality.
- **Fully Connected Layers**: Classify the image into one of seven emotion categories.
- **Output**: Softmax layer with 7 classes.

DATASET LINK
https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
