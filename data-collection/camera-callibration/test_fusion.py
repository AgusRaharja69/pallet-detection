# import cv2
# import numpy as np
# from rplidar import RPLidar

# # Calibration parameters - adjust according to your setup
# lidar_height = 10.5  # cm
# camera_height = 14.5  # cm
# image_width = 640
# image_height = 480

# # Capture and calibrate the camera
# cap = cv2.VideoCapture(1)
# ret, chessboard = cap.read()
# chessboard_size = (7, 5)  # Number of inner corners in the pattern
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# objpoints = []  # 3D points in the real-world coordinate system
# imgpoints = []  # 2D points in the camera image coordinate system

# # Generate real-world coordinate points for the calibration pattern
# objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# for i in range(5):
#     if ret==True:
#         gray = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)
#         ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
#         if ret:
#             objpoints.append(objp)
#             imgpoints.append(corners)

#         ret, chessboard = cap.read()

# # Calibrate the camera
# ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
#     objpoints, imgpoints, gray.shape[::-1], None, None)

# # Create a window to display the camera and lidar data
# cv2.namedWindow('Fused Image')

# # Connect to the Lidar and start scanning
# PORT_NAME = 'COM14'
# BAUDRATE: int = 115200
# lidar = RPLidar(PORT_NAME, baudrate=BAUDRATE, timeout=3)
# for scan in lidar.iter_scans(min_len=100, scan_type='normal', max_buf_meas=False):
#     # Extract the scan data and transform it to the camera coordinate system
#     points = []
#     for (_, angle, distance) in scan:
#         x = distance * np.sin(np.radians(angle)) # transform lidar data to cartesian (x,y)
#         y = distance * np.cos(np.radians(angle)) # transform lidar data to cartesian (x,y)
#         pos = np.array([x, y, lidar_height, 1]) # add height to align the data 
#         pos = np.dot(np.linalg.inv(tvecs), pos) # transform lidar data to camera coordinate system using camera extrinsics
#         pos = np.dot(np.linalg.inv(camera_matrix), pos) # transform lidar data to image plane
#         pos = pos / pos[2] # convert to homogeneous coordinates
#         x, y = int(pos[0]), int(pos[1])
#         if 0 <= x < image_width and 0 <= y < image_height:
#             points.append((x, y))

#     # Capture an image from the webcam and overlay the lidar data
#     ret, img = cap.read()
#     img = cv2.drawContours(img, [np.array(points)], -1, (255, 0, 0), 2)

#     # Display the image
#     cv2.imshow('Fused Image', img)
#     cv2.waitKey(1)

# cap.release()
# lidar.stop()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from rplidar import RPLidar

PORT_NAME = 'COM14'
BAUDRATE: int = 115200
lidar = RPLidar(PORT_NAME, baudrate=BAUDRATE, timeout=3)

# Calibration parameters - adjust according to your setup
lidar_height = 10.5  # cm
camera_height = 14.5  # cm
image_width = 640
image_height = 480

# Capture and calibrate the camera
cap = cv2.VideoCapture(1)
ret, chessboard = cap.read()
chessboard_size = (7, 5)  # Number of inner corners in the pattern
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []  # 3D points in the real-world coordinate system
imgpoints = []  # 2D points in the camera image coordinate system

# Generate real-world coordinate points for the calibration pattern
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

for i in range(1):
    if ret==True:
        gray = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

        ret, chessboard = cap.read()

# Calibrate the camera
ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# Calculate the Lidar to camera transformation matrix
rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
translation_vector = tvecs[0][0] / 10.0 + np.array([0, camera_height - lidar_height, 0])
extrinsic_matrix = np.concatenate((rotation_matrix, translation_vector.reshape(3, 1)), axis=1)
extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0, 1])))
camera_to_lidar = np.linalg.inv(extrinsic_matrix)

# Create a window to display the camera and Lidar data
cv2.namedWindow('Fused Image', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Fused Image', 640, 480) # optional - adjust the size if needed

while True:
    # Capture an image from the webcam
    ret, img = cap.read()
    # Extract the scan data and transform it to the camera coordinate system
    points = []
    # for scan in lidar.iter_scans(min_len=250, scan_type='normal', max_buf_meas=False):
    #     for (_, angle, distance) in scan:
    #         print(angle)
    #         pos = np.array([distance * np.sin(np.radians(angle)), -distance * np.cos(np.radians(angle)), 0, 1])
    #         pos = np.dot(camera_to_lidar, pos)
    #         pos = np.dot(camera_matrix, pos[:3])
    #         pos = pos / pos[2]
    #         x, y = int(pos[0]), int(pos[1])
    #         if 0 <= x < image_width and 0 <= y < image_height:
    #             points.append((x, y))

    #     # Overlay the Lidar data on the camera frame
    #     print("points: ",points)
    #     img = cv2.drawContours(img, [np.array(points)], -1, (255, 0, 0), 2)

    # Display the image
    cv2.imshow('Fused Image', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
lidar.stop()
cv2.destroyAllWindows()
