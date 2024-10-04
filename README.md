
# Camera and Robot Contour Tracking Project

This repository contains a set of scripts for camera calibration and real-time robot contour tracking using a Universal Robots (UR) robot arm. The system captures images through a camera, processes them to detect contours, and sends commands to the robot to track the detected contours.

## Files Description

- **camera_calibration.py**: This script performs camera calibration using a chessboard pattern to calculate the camera matrix and distortion coefficients.
  
- **test_camera_calibration.py**: A script to test the calibration data by capturing an image from the camera and applying the calibration parameters to undistort the image.

- **test_camera_robot_distances.py**: This script calculates the distances between a robot and detected objects using the calibrated camera data.

- **test_contour_tracking.py**: A script for detecting and tracking contours in an image using camera calibration data.

- **get_image_corners.py**: This script is used to capture and map the real-world coordinates of four corners of an image displayed in a camera feed. The user manually moves a robot to these corners while in "free drive" mode, and the script records the corresponding real-world coordinates. The coordinates are saved to a JSON file for future use in tasks requiring spatial calibration.

- **robot_contour_tracking.py**: The core script for real-time contour tracking with a UR robot. It captures images, detects contours, converts them into real-world coordinates, and sends movement commands to the robot.


## Publication

The initial details of this project were presented in the paper:

Busia, P., Marche, C., Meloni, P., & Reforgiato Recupero, D. (2024, June). **Design of an AI-driven Architecture with Cobots for Digital Transformation to Enhance Quality Control in the Food Industry**. In *Adjunct Proceedings of the 32nd ACM Conference on User Modeling, Adaptation and Personalization* (pp. 424-428).

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.

