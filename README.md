# Pizza-Box Stain Detection and Robot Trajectory Execution

This repository contains the code developed in WP4 for the **pizza-box waste disposal prototype**, which is composed of two main parts:

1. **Stain segmentation on pizza boxes**  
   Detection of stained areas on pizza boxes using a segmentation pipeline and grayscale processing, with contour extraction and simplification.

2. **Robot control and image-to-world mapping**  
   Conversion of image contours into real-world waypoints for a collaborative robot (UR), using camera calibration, ArUco-based calibration of the cardboard plane, and recorded contact heights to generate wiping trajectories.

This codebase serves as the reference implementation for the methods described in the WP4 deliverable (segmentation + robot control).

---

## Repository Structure

- `main.py`  
  End-to-end script that:
  - acquires an image from the camera  
  - loads camera calibration and image-to-robot-plane calibration  
  - performs pizza-box and stain segmentation  
  - refines stained regions via grayscale thresholding and dilation  
  - extracts and simplifies contours  
  - maps contour points to robot-frame coordinates via homography  
  - commands the UR robot to execute trajectories over the selected contours.

- `calibrazione_camera.py`  
  Camera calibration script using a chessboard pattern.  
  It captures calibration images from the webcam, estimates the intrinsic matrix and distortion coefficients, and saves them to `calibration_data.json`.

- `get_image_corners_aruco.py`  
  Script for computing image-to-world correspondences using ArUco markers.  
  It:
  - detects ArUco markers in the camera image  
  - lets the user move the robot in freedrive mode above each marker center  
  - reads the robot TCP pose and associates it with the marker’s pixel coordinates  
  - stores the resulting correspondences in `image_corners_real_coords_aruco.json`.

- `get_contact_point.py`  
  Utility for measuring the working height (`Z`) of the robot on the pizza-box plane.  
  It:
  - allows the user to move the tool in freedrive to the cardboard surface  
  - records the current TCP pose  
  - saves the contact height (and XY) into `z_heights_contact_points.json`.

- `make_photo.py`  
  Simple script to capture and display a single frame from the camera, useful to check framing, lighting, and focus.

- `calibration_data.json`  
  Example camera calibration file containing:
  - camera matrix  
  - distortion coefficients.

- `image_corners_real_coords_aruco.json`  
  Example output listing image coordinates of ArUco markers and corresponding robot-frame coordinates on the pizza-box plane.

- `z_heights_contact_points.json`  
  Example file storing one or more TCP poses used to set the contact height on the cardboard surface.

- `LICENSE.txt`  
  Project license information (Apache 2.0).

- `README.md`  
  This documentation file.

---

## Setup Guide

Setup instructions to install and configure the environment.

### Prerequisites

- **Python Version**: 3.9.11  
  Ensure you are using Python 3.9.11 as the specified version for compatibility.

- A **Universal Robots** arm reachable over the network, with RTDE enabled and the correct IP configured in the scripts.

- A **USB camera** (or equivalent) connected to the machine running the code.

### Setup Instructions

1. **Install Dependencies**

   From the project root, install all dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Then install additional required packages:

   ```bash
   pip install git+https://github.com/openai/CLIP.git
   pip install git+https://github.com/facebookresearch/segment-anything.git
   pip install roboflow supervision jupyter_bbox_widget
   ```

2. **Download Weights**

   Create a directory to store the model weights and download the required files:

   ```bash
   mkdir -p weights
   wget -P weights https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt
   wget -P weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

   By default, the scripts expect:

   - `weights/FastSAM.pt`  
   - `weights/sam_vit_h_4b8939.pth`

   If you change the location of these files, update the corresponding paths in `main.py`.

---

## General Workflow and How to Run

The typical workflow is:

1. **Set robot IP and check safety**

   - Edit the robot IP (e.g., `ROBOT_HOST`) in:
     - `main.py`
     - `get_image_corners_aruco.py`
     - `get_contact_point.py`
   - Make sure the robot is in a safe environment and that you are allowed to control it via RTDE.

2. **Calibrate the camera** (once per camera/setup)

   - Print a chessboard pattern with the square size specified in `calibrazione_camera.py`.
   - Run:

     ```bash
     python calibrazione_camera.py
     ```

   - Follow the on-screen instructions to capture calibration images.
   - The script will save `calibration_data.json` containing the intrinsic matrix and distortion coefficients.

3. **Compute image-to-world correspondences with ArUco markers**

   - Place four ArUco markers on the cardboard plane so that they are visible in the camera view.
   - Run:

     ```bash
     python get_image_corners_aruco.py
     ```

   - For each detected marker:
     - move the robot in freedrive mode to align the tool tip with the marker center  
     - confirm in the console when the pose is correct.
   - The script will save `image_corners_real_coords_aruco.json` with the pixel ↔ robot-frame correspondences, later used to compute the homography.

4. **Record the contact height**

   - Run:

     ```bash
     python get_contact_point.py
     ```

   - Move the robot in freedrive until the tool lightly touches the pizza-box surface.
   - Save the pose when prompted; the script will write `z_heights_contact_points.json`, which `main.py` uses to set the correct Z during wiping.

5. **Test the camera framing**

   - Run:

     ```bash
     python make_photo.py
     ```

   - Check that:
     - the pizza box is fully visible  
     - lighting is acceptable  
     - the resolution matches the one used for calibration and segmentation.

6. **Run the full segmentation + robot-control pipeline**

   - Place the pizza box in the calibrated area.
   - Ensure the robot is at a safe starting configuration and that the workspace is clear.
   - Run:

     ```bash
     python main.py
     ```

   - The script will:
     - capture an image  
     - detect the box and stained regions  
     - extract and simplify contours  
     - map them to robot coordinates  
     - execute a trajectory along the selected contours on the cardboard surface.

---

## Publication

The initial details of this project were presented in the paper:

Busia, P., Marche, C., Meloni, P., & Reforgiato Recupero, D. (2024, June). **Design of an AI-driven Architecture with Cobots for Digital Transformation to Enhance Quality Control in the Food Industry**. In *Adjunct Proceedings of the 32nd ACM Conference on User Modeling, Adaptation and Personalization* (pp. 424–428).

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.
