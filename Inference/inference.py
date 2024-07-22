# code for inference just pass path of image in line no. 55


import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('my_model_updated.h5', compile=False)

def predict_image(model, image_path, img_size=(224, 224), classes=None):
    # Load image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None

    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    # Resize image
    resized_img = cv2.resize(gray_img, img_size)
    
    # Normalize pixel values to range [0, 1]
    normalized_img = resized_img.astype('float32') / 255.0
    
    # Reshape to match model input shape
    input_img = normalized_img.reshape(1, img_size[0], img_size[1], 1)
    
    # Make prediction
    predictions = model.predict(input_img)
    print(predictions)
    
    # Get the predicted class
    predicted_class_index = np.argmax(predictions[0])
    print(predicted_class_index)
    
    # Get the class name
    if classes:
        predicted_class_name = classes[predicted_class_index]
    else:
        predicted_class_name = f"Class {predicted_class_index}"
    
    # Get the confidence score
    confidence = predictions[0][predicted_class_index]
    
    return predicted_class_name, confidence

# Example usage
classes = ['soccer_shoes', 'sandals', 'flip_flops', 'loafers', 'boots', 'sneakers']
image_path = "C:\\dataset\\shoeTypeClassifierDataset\\training\\soccer_shoes\\aug_2_image5.jpg"

predicted_class, confidence = predict_image(model, image_path, classes=classes)
if predicted_class is not None:
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")