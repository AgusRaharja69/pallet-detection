import cv2
import numpy as np

# Load the images of the checkerboard
images = []
for i in range(3):
    image = cv2.imread("saved_img_{}.jpg".format(i))
    images.append(image)

# Define the size of the checkerboard pattern
# pattern_size = (9, 6)
pattern_size = (7, 5)

# Detect the corners of the checkerboard pattern in each image
object_points = []
image_points = []
for image in images:
    found, corners = cv2.findChessboardCorners(image, pattern_size)
    if found:
        object_points.append(np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32))
        object_points[-1][:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        image_points.append(corners)

# Define the intrinsic parameters of the camera
image_size = (640, 480)
focal_length = image_size[0] / (2 * np.tan(55 / 2 * np.pi / 180))
principal_point = (image_size[0] / 2, image_size[1] / 2)
intrinsic_parameters = np.array([[focal_length, 0, principal_point[0]],
                                 [0, focal_length, principal_point[1]],
                                 [0, 0, 1]])

# Estimate the extrinsic parameters of the camera
_, cameraMatrix, dist, rotation_vectors, translation_vectors = cv2.calibrateCamera(object_points, image_points, image_size, intrinsic_parameters, None)

# Extract the rotation and translation matrices from the estimated parameters
rotation_matrices = []
for rotation_vector in rotation_vectors:
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    rotation_matrices.append(rotation_matrix)
translation_matrices = translation_vectors

print(" Camera matrix:")
print(cameraMatrix)

print("\n Distortion coefficient:")
print(dist)

print("\n Rotation Vectors:")
print(rotation_matrices)

print("\n Translation Vectors:")
print(translation_matrices)



import numpy as np
import cv2

# Collect data from both sensors
camera_images = []  # replace with actual camera images

# Camera calibration
chessboard_size = (9, 6)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], camera_images, (640, 480), None, None)

# Compute camera matrix from diagonal field of view
diagonal_fov = 55.0  # degrees
fx = fy = (640 / 2) / np.tan(np.radians(diagonal_fov / 2))
K = np.array([[fx, 0, 640 / 2], [0, fy, 480 / 2], [0, 0, 1]])

print(k)

# # Hand-eye calibration
# poses_lidar = []  # replace with actual lidar poses
# poses_camera = []  # replace with actual camera poses
# T_lidar2cam = cv2.calibrateHandEye(poses_lidar, poses_camera, method=cv2.CALIB_HAND_EYE_TSAI)

# # Compute rotation and translation matrix from transformation matrix
# R_lidar2cam = T_lidar2cam[:3, :3]
# t_lidar2cam = T_lidar2cam[:3, 3]

# print("Rotation matrix from lidar to camera:")
# print(R_lidar2cam)
# print("Translation vector from lidar to camera:")
# print(t_lidar2cam)
