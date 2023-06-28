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

for i in range(5):
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

print("camera_matrix: ", camera_matrix)
print("rvecs: ", rvecs)
print("tvecs: ", tvecs)
print("rotation_matrix: ", rotation_matrix)
print("translation_vector: ", translation_vector)
print("extrinsic_matrix: ", extrinsic_matrix)
print("camera_to_lidar: ", camera_to_lidar)

# Create a window to display the camera and Lidar data
cv2.namedWindow('Fused Image', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Fused Image', 640, 480) # optional - adjust the size if needed

##########################
# import cv2
# import numpy as np
# import json

# jsonLidarPath = '../lidarJson.json'

# # Calibration parameters - adjust according to your setup
# lidar_height = 10.5  # cm
# camera_height = 14.5  # cm
# image_width = 640
# image_height = 480

# # Calibration data chessboard
# ########### benar ############
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

# #############################
# # fx = float(1570.34678)
# # fy = float(1498.80818)
# # cx = float(375.793671)
# # cy = float(365.519559)

# # r00 = float(0.93358714)
# # r01 = float(-0.01127525)
# # r02 = float(-0.35817302)
# # r10 = float(-0.02131085)
# # r11 = float(0.995989)
# # r12 = float(-0.0869009)
# # r20 = float(0.35771622)
# # r21 = float(0.08876254)
# # r22 = float(0.92960224)

# # t0 = float(-0.47187442)
# # t1 = float(3.52812558)
# # t2 = float(-0.47187442)



# # Calibration data obtained from the previous code
# camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# rotation_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
# translation_vector = np.array([t0, t1, t2])

# # Calculate camera-to-lidar transformation matrix
# rotation_matrix_inv = np.linalg.inv(rotation_matrix)
# translation_vector_inv = -rotation_matrix_inv.dot(translation_vector)
# extrinsic_matrix_inv = np.concatenate((rotation_matrix_inv, translation_vector_inv.reshape(3, 1)), axis=1)
# extrinsic_matrix_inv = np.vstack((extrinsic_matrix_inv, np.array([0, 0, 0, 1])))

# # Create VideoCapture object
# cap = cv2.VideoCapture(1)  # Adjust camera index if needed

# while True:
#     # Read frame from the camera
#     ret, frame = cap.read()
    
#     # Lidar data
#     # Lidar points in polar coordinates
#     with open(jsonLidarPath) as f:
#         try :
#             lidar_data = json.load(f)
#         except:
#             lidar_data = {"data": [[0,0,0]]}

#     # LiDAR data
#     dataLidar = np.array(lidar_data['data'])
#     lidar_angles = np.radians(dataLidar[:, 1]) # Convert angle to radians
#     lidar_distances = dataLidar[:, 2] #lidar distance
#     lidar_angles = 1.5*np.pi - lidar_angles #lidar angle

#     # Project lidar data into camera frame
#     lidar_points = np.zeros((len(lidar_distances), 3))
#     for i in range(len(lidar_distances)):
#         distance = lidar_distances[i]/100
#         angle = lidar_angles[i]

#         x = distance * np.cos(angle)
#         y = distance * np.sin(angle)
#         z = lidar_height

#         lidar_points[i] = [x, y, z]

#     lidar_points_homogeneous = np.hstack((lidar_points, np.ones((len(lidar_points), 1))))
#     camera_points_homogeneous = extrinsic_matrix_inv.dot(lidar_points_homogeneous.T).T

#     # Convert camera points to image coordinates
#     camera_points = camera_points_homogeneous[:, :3] / camera_points_homogeneous[:, 3, np.newaxis]
#     image_points, _ = cv2.projectPoints(camera_points, np.zeros((3,)), np.zeros((3,)), camera_matrix, None)

#     # print(image_points)
#     # Display projected lidar points on the camera image
#     for i in range(len(image_points)):
#         x, y = image_points[i].ravel()
#         print([x,y,np.degrees(lidar_angles[i])])
#         cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
#     # for point in image_points:
#     #     x, y = point.ravel()
#     #     print([x,y])
#     #     cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

#     # Show the frame with projected lidar points
#     cv2.imshow('Projected Lidar Points', frame)
    
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the VideoCapture object and close any open windows
# cap.release()
# cv2.destroyAllWindows()

