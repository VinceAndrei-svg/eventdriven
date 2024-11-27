import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def load_data(dataset_path):
    X, y = [], []
    # Loop through the folders A, B, C, D
    for label in ['A', 'B', 'C', 'D']:
        folder_path = os.path.join(dataset_path, label)
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            # Load image in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Resize the image to 50x50 pixels to standardize the input size
                resized_image = cv2.resize(image, (50, 50)).flatten()
                X.append(resized_image)
                y.append(label)
    return np.array(X), np.array(y)

if __name__ == '__main__':
    # Set the dataset path
    dataset_path = 'datasets'
    
    # Load the dataset
    X, y = load_data(dataset_path)

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Support Vector Classifier (SVC) model
    model = SVC(kernel='linear', probability=True)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the trained model to a file (bubble_model.pkl)
    joblib.dump(model, 'ml_model/bubble_model.pkl')
    print("Model saved as bubble_model.pkl")
