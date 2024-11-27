import cv2
import numpy as np
import joblib

def predict_answer(image_path):
    # Load the trained model
    model = joblib.load('ml_model/bubble_model.pkl')

    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 50x50 and flatten it into a feature vector
    resized_image = cv2.resize(image, (50, 50)).flatten().reshape(1, -1)

    # Predict the answer (A, B, C, or D)
    predicted_answer = model.predict(resized_image)[0]
    return predicted_answer
