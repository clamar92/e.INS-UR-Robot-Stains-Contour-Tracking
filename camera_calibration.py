import cv2
import numpy as np
import json
import os

# Chessboard dimensions (number of internal intersections)
chessboard_size = (9, 6)

# Size of each chessboard square in millimeters
square_size = 17  # mm

# Prepare object points with real-world dimensions in millimeters
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Convert to millimeters

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image space

# Number of calibration images to capture
num_calibration_images = 20

# Start the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the webcam is successfully opened
if not cap.isOpened():
    print("Error opening the webcam")
    exit()

captured_images = 0

while captured_images < num_calibration_images:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error reading the frame from the webcam")
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Refine the corner detections for better accuracy
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), 
            (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        )

        # Draw and display the chessboard corners
        frame = cv2.drawChessboardCorners(frame, chessboard_size, corners_refined, ret)
        cv2.imshow('Calibration', frame)
        
        captured_images += 1
        print(f"Captured calibration image: {captured_images}/{num_calibration_images}")
        
        # Wait for the user to press a key to capture the next image
        cv2.waitKey(0)
    else:
        cv2.imshow('Webcam', frame)
        cv2.waitKey(1)  # Brief pause to handle GUI events

# Release the webcam
cap.release()
cv2.destroyAllWindows()

# Check if any points were collected for calibration
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("No points found for calibration")
    exit()

# Perform camera calibration using the collected points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if not ret:
    print("Calibration failed")
    exit()

# Print the calibration matrix and distortion coefficients
print("Calibration matrix:")
print(mtx)
print("Distortion coefficients:")
print(dist)

# Calculate the reprojection error in millimeters
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)
print(f"Average reprojection error (in mm): {mean_error}")

# Save the calibration data to a JSON file
output_dir = ''
os.makedirs(output_dir, exist_ok=True)
calibration_data_path = os.path.join(output_dir, 'calibration_data.json')

calibration_data = {
    'mtx': mtx.tolist(),
    'dist': dist.tolist(),
    'rvecs': [rvec.tolist() for rvec in rvecs],
    'tvecs': [tvec.tolist() for tvec in tvecs],
    'mean_error': mean_error
}

with open(calibration_data_path, 'w') as f:
    json.dump(calibration_data, f)

print(f"Calibration data saved to {calibration_data_path}")
