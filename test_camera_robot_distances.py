import cv2
import numpy as np
import json
import os

# Load the saved calibration data
calibration_data_path = 'calibration_data.json'
if not os.path.exists(calibration_data_path):
    print(f"Error: The file {calibration_data_path} does not exist. Please perform calibration first.")
    exit()

with open(calibration_data_path, 'r') as f:
    calibration_data = json.load(f)

# Extract calibration parameters
mtx = np.array(calibration_data['mtx'])
dist = np.array(calibration_data['dist'])

# Start the webcam to capture a photo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error opening the webcam")
    exit()

# Capture a single frame
ret, frame = cap.read()
if not ret:
    print("Error capturing image from the webcam")
    cap.release()
    exit()

# Save the captured image for testing
image_path = 'captured_image.jpg'
cv2.imwrite(image_path, frame)
print(f"Captured image saved as {image_path}")

# Release the webcam
cap.release()

# Read the captured image for undistortion
img = cv2.imread(image_path)

if img is None:
    print("Error loading the image.")
    exit()

# Get the image dimensions
h, w = img.shape[:2]

# Obtain a new camera matrix to correct the distortion
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Correct the distortion using the new camera matrix
undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_matrix)

# Crop the image using the ROI (Region of Interest)
x, y, w, h = roi
undistorted_img = undistorted_img[y:y + h, x:x + w]

# Show the original and undistorted images
cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', undistorted_img)

# Wait for user interaction before closing windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the undistorted image
undistorted_image_path = '13_09_2024/undistorted_image.jpg'
cv2.imwrite(undistorted_image_path, undistorted_img)
print(f"Undistorted image saved as {undistorted_image_path}")