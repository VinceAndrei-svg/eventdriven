import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

def segment_bubbles(image_path):
    """
    Segments the answer sheet into individual bubble images.
    Returns a list of file paths for each segmented bubble.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error loading the image.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours for circular bubbles
    bubble_contours = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if len(approx) > 8 and 100 < area < 1000:  # Adjust thresholds as needed
            bubble_contours.append(contour)

    # Sort contours by position (top-to-bottom, left-to-right)
    bubble_contours = sorted(bubble_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    # Save cropped bubble images
    cropped_images = []
    for i, contour in enumerate(bubble_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped = image[y:y+h, x:x+w]
        bubble_path = f"temp_bubble_{i}.png"
        cv2.imwrite(bubble_path, cropped)
        cropped_images.append(bubble_path)

    return cropped_images

def preprocess_image(image_path):
    """
    Preprocesses a single bubble image for model prediction.
    """
    img = Image.open(image_path).convert('RGB')  # Ensure RGB format
    img = img.resize((256, 256))  # Resize to model input size
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array
