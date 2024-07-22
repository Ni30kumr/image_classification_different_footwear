# Footwear Image Classification Project

## Task
This project aims to classify footwear images into six different categories: boots, flip-flops, loafers, sandals, sneakers, and soccer shoes.

## Dataset
The dataset is located at "C:\dataset\shoeTypeClassifierDataset\training" and contains images of different types of footwear, organized into subdirectories for each category.

### Dataset Structure
- boots
- flip_flops
- loafers
- sandals
- sneakers
- soccer_shoes

## Data Analysis and Preprocessing

### Image Format Analysis
A data_analysis_and_augmentation.ipynb was developed to analyze the image formats present in each category folder. This ensures that the data loading process can handle all present formats.

Key findings:
- Most categories (boots, flip-flops, loafers, sandals) contain images in .jpg, .jpeg, and .png formats.
- The sneakers category contains only .jpg images.
- The soccer_shoes category has the most diverse set of formats, including .jpg, .gif, .jpeg, and .png.

### Comprehensive Image Analysis
A data_analysis_and_augmentation script was created to perform an in-depth analysis of the image dataset. This script collects the following information for each class:

1. Total image count
2. Image sizes (average, min, max)
3. Color modes
4. File formats
5. Aspect ratios (average, min, max)
6. File sizes in KB (average, min, max)
7. Pixel statistics (mean and standard deviation for each color channel)

This analysis provides crucial insights into the dataset's characteristics, which can inform preprocessing steps and model architecture decisions.

### Face Detection and Image Cropping
To ensure the dataset focuses solely on footwear, a two-step cleaning process was implemented:

1. Face Detection:
   - Uses the `face_recognition` library to detect faces in images.
   - Images containing faces are moved to a separate folder for further processing.

2. Image Cropping:
   - Images with detected faces are cropped to focus on the footwear.
   - The bottom 40% of each image is retained, assuming this portion contains the footwear.
   - This process helps eliminate irrelevant parts of the image, focusing the dataset on the classification task.

Example usage:
```python
# Face detection
input_folder = "C:\\dataset\\shoeTypeClassifierDataset\\training\\soccer_shoes"
output_folder = "C:\\dataset\\shoeTypeClassifierDataset\\training\\soccer_shoes_with_faces"
move_images_with_faces(input_folder, output_folder)

# Image cropping
input_folder = r"C:\dataset\shoeTypeClassifierDataset\training\soccer_shoes_with_faces"
output_folder = r"C:\dataset\shoeTypeClassifierDataset\training\soccer_shoes_resize"
process_folder(input_folder, output_folder, keep_ratio=0.4)

### Dataset Preparation
The `prepare_dataset` function performs the following steps:
1. Loads images from the specified directory
2. Converts images to grayscale
3. Resizes images to 224x224 pixels
4. Normalizes pixel values to the range [0, 1]
5. Converts class labels to one-hot encoded format

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following structure:

1. Three convolutional blocks, each containing:
   - Two Conv2D layers with BatchNormalization
   - MaxPooling2D layer
2. Flatten layer
3. Two Dense layers with BatchNormalization and Dropout
4. Output Dense layer with softmax activation

Key features:
- Input shape: (224, 224, 1) (grayscale images)
- Total parameters: [Add the number of parameters from model.summary()]

## Model Training

Training parameters:
- Optimizer: Adam with learning rate of 0.001
- Loss function: Categorical crossentropy
- Metrics: Accuracy
- Batch size: 32
- Epochs: 30

The training process includes:
- Shuffling of training data
- Validation using a separate validation set
- Monitoring of training and validation accuracy/loss

## Model Evaluation

The model is evaluated using:
1. Test accuracy on the validation set
2. Classification report (precision, recall, f1-score)
3. Confusion matrix
4. Sample predictions on validation data

## Model Saving

The trained model is saved in H5 format:
```python
model.save('my_model_updated.h5')

## Inference

The project includes an inference script (`inference.py`) that allows for classification of individual footwear images using the trained model.

### Inference Process

1. The trained model is loaded from the saved H5 file.
2. The input image is preprocessed:
   - Converted to grayscale
   - Resized to 224x224 pixels
   - Normalized to pixel values in the range [0, 1]
3. The preprocessed image is fed into the model for prediction.
4. The script returns the predicted class name and confidence score.

### Usage

To use the inference script:

1. Ensure the trained model file `my_model_updated.h5` is in the same directory as the script.
2. Run the script and provide the path to the image you want to classify:

```python
python inference.py
