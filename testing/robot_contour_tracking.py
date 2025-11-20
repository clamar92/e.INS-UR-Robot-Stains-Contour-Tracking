import cv2
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.measure import find_contours
import rtde_control
import rtde_receive
import math

# Robot IP (set to your specific robot's IP)
ROBOT_HOST = '192.168.137.198'

# Initialize RTDE control interfaces (disabled for now)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Robot motion parameters (acceleration and speed)
acc = 0.4
vel = 0.4

# Initial robot position in radians
robot_startposition = [math.radians(17.87),
                       math.radians(-78.87),
                       math.radians(-100.97),
                       math.radians(-90.22),
                       math.radians(90.03),
                       math.radians(15.62)]

# Define robot motion speed and direction
speed = [0, 0, -0.1, 0, 0, 0]
direction = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
acceleration = 1  # Acceleration

# Move robot to initial position
rtde_c.moveJ(robot_startposition, vel, acc)

# Image resolution used during calibration
img_width = 640
img_height = 480

# Path to the calibration data files
calibration_file = 'image_corners_real_coords.json'
camera_calibration_file = 'calibration_data.json'

# Load the camera calibration parameters
with open(camera_calibration_file, 'r') as f:
    calib_data = json.load(f)
    camera_matrix = np.array(calib_data['mtx'])
    dist_coeffs = np.array(calib_data['dist'])

# Load the calibration points (image points -> real-world robot points)
with open(calibration_file, 'r') as f:
    calibration_data = json.load(f)

# Extract image points and real-world points from the file
image_points = np.array([point['corner'] for point in calibration_data], dtype=np.float32)
real_points = np.array([point['real'][:2] for point in calibration_data], dtype=np.float32)  # Use only X and Y

# Find the minimum and maximum real-world coordinates for X and Y
real_x_min, real_y_min = np.min(real_points, axis=0)
real_x_max, real_y_max = np.max(real_points, axis=0)

# Compute the homography matrix (image -> real-world)
homography_matrix, _ = cv2.findHomography(image_points, real_points)

# Function to map image coordinates (pixels) to real-world coordinates (robot)
def map_image_to_real(image_point):
    # Add a third coordinate for the homography transformation
    image_point_homog = np.array([image_point[0], image_point[1], 1], dtype=np.float32)
    
    # Apply the homography matrix
    real_point_homog = np.dot(homography_matrix, image_point_homog)
    
    # Normalize to get real-world coordinates (X, Y)
    real_x = real_point_homog[0] / real_point_homog[2]
    real_y = real_point_homog[1] / real_point_homog[2]
    
    return real_x, real_y

# Function to check if the real-world coordinates are within defined bounds
def is_within_bounds(real_x, real_y):
    return real_x_min <= real_x <= real_x_max and real_y_min <= real_y <= real_y_max

# Function to move the robot to a real-world point
def move_robot_to_real_point(real_x, real_y, first_mov, real_z=0.05):
    if is_within_bounds(real_x, real_y):
        print(f"Calculated real-world coordinates: X={real_x}, Y={real_y}, Z={real_z}")

        # Get the current TCP (tool center point) pose
        center = rtde_r.getActualTCPPose()
        center[0] = real_x
        center[1] = real_y

        if first_mov == 1:
            rtde_c.moveUntilContact(speed, direction, acceleration)
        else:
            rtde_c.moveL(center, 0.2, 0.2)

    else:
        print(f"Coordinates out of bounds: X={real_x}, Y={real_y}")

# Start the webcam and capture a single image
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

if not cap.isOpened():
    print("Error opening the webcam")
    exit()

# Capture a single frame
ret, frame = cap.read()

if not ret:
    print("Error reading the frame from the webcam")
else:
    # Correct the optical distortion using calibration parameters
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (img_width, img_height), 1)
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Convert the image to HSV and use the saturation channel for contour detection
    myimg_hsv = rgb2hsv(undistorted_frame)
    saturation = myimg_hsv[:, :, 1]  # Saturation channel

    # Create a binary image based on a saturation threshold
    binary_image = np.where(saturation > 0.25, 1, 0).astype(np.uint8)

    # Find contours based on the binary image
    contours_gray = find_contours(binary_image, 0.8)

    # If contours are found, select the largest contour
    if contours_gray:
        best_contour = max(contours_gray, key=len)

        # Visualize the images and contours
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))

        # Show the original image
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original Image')

        # Convert the contour to a format compatible with OpenCV and draw it on the image
        best_contour_cv2 = np.array([[[int(p[1]), int(p[0])]] for p in best_contour])
        cv2.drawContours(undistorted_frame, [best_contour_cv2], -1, (0, 255, 0), 2)
        ax[1].imshow(cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Image with Longest Contour')

        plt.show()

        # Convert the contour to real-world coordinates
        real_coords = []
        for i, point in enumerate(best_contour):
            image_point = np.array([[point[1], point[0]]], dtype='float32')  # Image plane coordinates (x, y)

            # Map the image point to the robot's real-world coordinates
            real_x, real_y = map_image_to_real(image_point[0])
            
            # Print the real-world coordinates
            if i == 0:
                move_robot_to_real_point(real_x, real_y, 1)
            else:
                move_robot_to_real_point(real_x, real_y, 0)

            # Add the real-world coordinates to the list
            real_coords.append([real_x, real_y])

        real_coords = np.array(real_coords)

        # Visualize the original image, the contour, and the real-world coordinates
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        # Show the original image
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original Image')

        # Draw the contour on the undistorted image
        best_contour_cv2 = np.array([[[int(p[1]), int(p[0])]] for p in best_contour])
        cv2.drawContours(undistorted_frame, [best_contour_cv2], -1, (0, 255, 0), 2)
        ax[1].imshow(cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Image with Longest Contour')

        # Plot the real-world coordinates on a 2D graph
        ax[2].plot(real_coords[:, 0], real_coords[:, 1], 'ro-')
        ax[2].set_xlabel('X (bottom to top)')
        ax[2].set_ylabel('Y (right to left)')
        ax[2].invert_xaxis()  # Invert the X-axis to simulate bottom-to-top movement
        ax[2].invert_yaxis()  # Invert the Y-axis to simulate right-to-left movement

        # Set axis limits based on real-world coordinates
        ax[2].set_xlim(real_x_min, real_x_max)
        ax[2].set_ylim(real_y_min, real_y_max)

        plt.tight_layout()
        plt.show()

    else:
        print("No contours found.")

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Move the robot back to the start position after processing
time.sleep(2)
print('Move robot to start position')
rtde_c.moveJ(robot_startposition, vel, acc)
