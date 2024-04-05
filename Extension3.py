"""
Saikiran Juttu | Kirti Kshirsagar | Project-5 | 4th April 2024 | Spring 2024
This file represents Extension(Build a live video digit recognition application using the trained network)of the assignment.
Here, we are using the pretrained model from task1 and running it on live video to predict the digits displayed.
"""
import cv2
import torch
import torch.nn.functional as F
from Task1ABCD import MyNetwork  # Import your trained network class
import numpy as np

# Load trained model
model = MyNetwork()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))  # Resize to match MNIST image size
    normalized = resized / 255.0  # Normalize pixel values

    # Convert numpy array to PyTorch tensor
    tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Perform digit recognition
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()

    # Display the frame with predicted digit
    cv2.putText(frame, str(prediction), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('Live Digit Recognition', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
