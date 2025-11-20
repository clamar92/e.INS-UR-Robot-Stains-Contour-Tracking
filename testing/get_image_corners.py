import cv2
import numpy as np
import time
import threading
import math
import rtde_control
import rtde_receive
import json

# Robot IP address and RTDE port
ROBOT_HOST = '192.168.137.198'
#ROBOT_HOST = '192.168.186.135'

# Corner points saved in the image_corners_real_coords.json file
# defined in the following order: top-left, top-right, bottom-left, bottom-right

# RTDE initialization
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Global variables for coordinates
corner_points = []
robot_moving = False
free_drive_active = False

# Speed and acceleration parameters
acc = 0.4
vel = 0.4

# Initial robot position
robot_startposition = [math.radians(0),
                       math.radians(-95),
                       math.radians(-100),
                       math.radians(-78),
                       math.radians(88),
                       math.radians(0)]

def move_to_start_position():
    """
    Move the robot to the predefined start position.
    """
    print('Move robot to start position')
    rtde_c.moveJ(robot_startposition, vel, acc)
    time.sleep(2)  # Wait until the movement is complete

def free_drive_mode():
    """
    Enable free drive mode, allowing the user to manually move the robot to the desired position.
    """
    global robot_moving, free_drive_active
    free_drive_active = True
    print("Free drive mode enabled. Move the robot to the desired position and press 'q' to save the coordinates.")
    rtde_c.teachMode()
    while free_drive_active:
        time.sleep(0.1)
    rtde_c.endTeachMode()
    robot_moving = False

def capture_real_coordinates():
    """
    Capture the real-world coordinates from the robot's current TCP (Tool Center Point) pose.
    """
    pose = rtde_r.getActualTCPPose()
    real_x, real_y, real_z = pose[:3]
    return real_x, real_y, real_z

def save_corner_point(corner_point, real_point):
    """
    Save the image corner point and its corresponding real-world coordinates.
    """
    corner_points.append({'corner': corner_point, 'real': real_point})

def main():
    global robot_moving, free_drive_active

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error opening the webcam")
        return

    # Image corners (top-left, top-right, bottom-left, bottom-right)
    image_corners = [(0, 0), (639, 0), (0, 479), (639, 479)]

    # Move the robot to the start position
    move_to_start_position()

    for i, corner in enumerate(image_corners):
        print(f"Move the robot to the image corner: {corner}. Press 'q' to save this point.")
        threading.Thread(target=free_drive_mode).start()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error capturing the frame")
                break

            cv2.circle(frame, corner, 10, (0, 255, 0), 2)
            cv2.putText(frame, f"Move the robot to corner: {corner}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Coordinate Capture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                free_drive_active = False
                robot_moving = True
                break

        real_point = capture_real_coordinates()
        save_corner_point(corner, real_point)
        print(f"Point {i+1} saved: Image {corner}, Real {real_point}")

        # Move the robot back to the start position
        time.sleep(2)
        move_to_start_position()

    cap.release()
    cv2.destroyAllWindows()

    # Save the captured corner points and their real-world coordinates to a JSON file
    with open('19_07_24/image_corners_real_coords.json', 'w') as f:
        json.dump(corner_points, f)

    print("Real-world coordinates for the 4 image corners saved.")

if __name__ == "__main__":
    main()
