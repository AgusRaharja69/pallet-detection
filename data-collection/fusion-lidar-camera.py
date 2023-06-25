# import cv2
# import numpy as np
# import json

# jsonLidarPath = '../lidarJson.json'

# # Calibration parameters - adjust according to your setup
# lidar_height = 10.5  # cm
# camera_height = 14.5  # cm
# image_width = 640
# image_height = 480

# fx = float(1570.34678)
# fy = float(1498.80818)
# cx = float(375.793671)
# cy = float(365.519559)

# camera_matrix = np.matrix(
#                 [[fx, 0.0, cx],
#                 [0.0, fy, cy],
#                 [0.0, 0.0, 1.0]])

# camera_to_lidar = np.matrix(
#             [[ 0.93358714, -0.02131085, 0.35771622, 0.68452038],
#             [-0.01127525, 0.995989, 0.08876254, -3.47740999],
#             [-0.35817302, -0.0869009, 0.92960224, 0.57624013],
#             [ 0.0, 0.0, 0.0, 1.0]])

# # Capture and calibrate the camera
# cap = cv2.VideoCapture(1)

# while True:
#     # Capture frame from webcam
#     ret, frame = cap.read()

#     # Lidar points in polar coordinates
#     with open(jsonLidarPath) as f:
#         try :
#             lidar_data = json.load(f)
#         except:
#             lidar_data = {"data": [[0,0,0]]}

#     # LiDAR data
#     dataLidar = np.array(lidar_data['data'])
#     angleRawLidar = np.radians(dataLidar[:, 1]) # Convert angle to radians
#     distance = dataLidar[:, 2] #lidar distance
#     angle = np.pi - angleRawLidar #lidar angle

#     # Convert polar coordinates to 3D cartesian coordinates in lidar frame
#     x = distance * np.sin(angle)
#     y = -distance * np.cos(angle)
#     z = np.zeros_like(x)
#     lidar_pos = np.column_stack([x, y, z, np.ones_like(distance)])

#     # Transform lidar points to camera frame
#     camera_pos = np.dot(camera_to_lidar, lidar_pos.T).T

#     # Remove homogeneous coordinate
#     camera_pos = camera_pos[:, :3] / camera_pos[:, 3:]

#     # Project camera 3D points to 2D image plane
#     pixel_pos = np.dot(
#         camera_matrix, 
#         np.dot(camera_to_lidar[:3,:3], 
#                np.vstack((distance * np.cos(angle), 
#                           -distance * np.sin(angle), 
#                           np.zeros_like(distance), 
#                           np.ones_like(distance)))))

#     # Extract x and y coordinates from pixel_pos
#     x, y = (pixel_pos[:-1, :]/pixel_pos[-1:, :]).T

#     # Create a list of points to draw contours
#     pts = np.array([(x[i], y[i]) for i in range(len(distance))], dtype=np.int32)

#     # Draw contours on the frame
#     cv2.drawContours(frame, [pts], 0, (0, 0, 255), 2)

#     # Display the resulting frame
#     cv2.imshow('frame', frame)

#     # Exit the program on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()

#######################################
# import numpy as np

# # Camera matrix
# camera_matrix = np.array([[1.57034678e+03, 0.00000000e+00, 3.75793671e+02],
#                           [0.00000000e+00, 1.49880818e+03, 3.65519559e+02],
#                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# # Camera to lidar transform
# camera_to_lidar = np.array([[0.93358714, -0.02131085, 0.35771622, 0.68452038],
#                             [-0.01127525, 0.995989, 0.08876254, -3.47740999],
#                             [-0.35817302, -0.0869009, 0.92960224, 0.57624013],
#                             [0., 0., 0., 1.]])

# # Lidar point in polar coordinates
# distance = 10  # meters
# angle = 45  # degrees

# # Convert polar coordinates to 3D cartesian coordinates in lidar frame
# x = distance * np.sin(np.radians(angle))
# y = -distance * np.cos(np.radians(angle))
# z = 0
# lidar_pos = np.array([x, y, z, 1])

# # Transform lidar point to camera frame
# camera_pos = np.dot(camera_to_lidar, lidar_pos)

# # Remove homogeneous coordinate
# camera_pos = camera_pos / camera_pos[3]

# # Project camera 3D point to 2D image plane
# pixel_pos = np.dot(camera_matrix, camera_pos[:3])

# # Normalize pixel_pos to get the (x, y) coordinates of the pixel
# x, y = pixel_pos[:2] / pixel_pos[2]

# # Print results
# print("Lidar point in polar coordinates:", (distance, angle))
# print("Lidar point in cartesian coordinates (lidar frame):", (x, y, z))
# print("Lidar point in cartesian coordinates (camera frame):", tuple(camera_pos[:3]))
# print("Pixel coordinates:", (x, y))

##########################
import cv2
import numpy as np
import json

jsonLidarPath = '../lidarJson.json'

# Calibration parameters - adjust according to your setup
lidar_height = 10.5  # cm
camera_height = 14.5  # cm
image_width = 640
image_height = 480

# Calibration data chessboard
# fx = float(437.81464946)
# fy = float(421.31344926)
# cx = float(250.51843603)
# cy = float(242.87899323)

# r00 = float(0.99216097)
# r01 = float(-0.00893324)
# r02 = float(-0.1246467)
# r10 = float(0.00491244)
# r11 = float(0.99945876)
# r12 = float(-0.03252774)
# r20 = float(0.12486981)
# r21 = float(0.03166044)
# r22 = float(0.99166786)

# t0 = float(-0.18843171)
# t1 = float(3.81156829)
# t2 = float(0.99166786)

# Based on point Lidar to camera
fx = float(6826.66667)
fy = float(6826.66667)
cx = float(320)
cy = float(240)

r00 = float(1)
r01 = float(0)
r02 = float(0)
r10 = float(0)
r11 = float(1)
r12 = float(0)
r20 = float(0)
r21 = float(0)
r22 = float(1)

t0 = float(0)
t1 = float(4)
t2 = float(0)

# Calibration data obtained from the previous code
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
rotation_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
translation_vector = np.array([t0, t1, t2])

# Calculate camera-to-lidar transformation matrix
rotation_matrix_inv = np.linalg.inv(rotation_matrix)
translation_vector_inv = -rotation_matrix_inv.dot(translation_vector)
extrinsic_matrix_inv = np.concatenate((rotation_matrix_inv, translation_vector_inv.reshape(3, 1)), axis=1)
extrinsic_matrix_inv = np.vstack((extrinsic_matrix_inv, np.array([0, 0, 0, 1])))

# Create VideoCapture object
cap = cv2.VideoCapture(1)  # Adjust camera index if needed

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    
    # Lidar data
    # Lidar points in polar coordinates
    with open(jsonLidarPath) as f:
        try :
            lidar_data = json.load(f)
        except:
            lidar_data = {"data": [[0,0,0]]}

    # LiDAR data
    dataLidar = np.array(lidar_data['data'])
    lidar_angles = np.radians(dataLidar[:, 1]-90) # Convert angle to radians
    lidar_distances = dataLidar[:, 2] #lidar distance
    # lidar_angles = 0.5*np.pi - lidar_angles #lidar angle

    # Project lidar data into camera frame
    lidar_points = np.zeros((len(lidar_distances), 3))
    for i in range(len(lidar_distances)):
        distance = lidar_distances[i]/100
        angle = lidar_angles[i]

        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        z = lidar_height

        lidar_points[i] = [x, y, z]

    lidar_points_homogeneous = np.hstack((lidar_points, np.ones((len(lidar_points), 1))))
    camera_points_homogeneous = extrinsic_matrix_inv.dot(lidar_points_homogeneous.T).T

    # Convert camera points to image coordinates
    camera_points = camera_points_homogeneous[:, :3] / camera_points_homogeneous[:, 3, np.newaxis]
    image_points, _ = cv2.projectPoints(camera_points, np.zeros((3,)), np.zeros((3,)), camera_matrix, None)

    print(image_points)
    # Display projected lidar points on the camera image
    for point in image_points:
        x, y = point.ravel()
        cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)

    # Show the frame with projected lidar points
    cv2.imshow('Projected Lidar Points', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close any open windows
cap.release()
cv2.destroyAllWindows()