import cv2
import numpy as np
import yaml

# Calibration parameters - adjust according to your setup
lidar_height = 10.5  # cm
camera_height = 14.5  # cm
image_width = 640
image_height = 480
chessboard_size = (7, 5)  # Number of inner corners in the pattern

# Capture and calibrate the camera
cap = cv2.VideoCapture(1)  # Adjust camera index if needed

objpoints = []  # 3D points in the real-world coordinate system
imgpoints = []  # 2D points in the camera image coordinate system

while len(objpoints) < 5:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', frame)
        cv2.waitKey(500)  # Delay to visualize the detected corners

cv2.destroyAllWindows()

# Calibrate the camera
ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (image_width, image_height), None, None)

# Calculate the Lidar to camera transformation matrix
rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
translation_vector = tvecs[0][0] / 10.0 + np.array([0, camera_height - lidar_height, 0])
extrinsic_matrix = np.concatenate((rotation_matrix, translation_vector.reshape(3, 1)), axis=1)
extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0, 1])))
camera_to_lidar = np.linalg.inv(extrinsic_matrix)

print("camera_matrix: ", camera_matrix)
print("rvecs: ", rvecs)
print("tvecs: ", tvecs)
print("dist: ", dist)
print("rotation_matrix: ", rotation_matrix)
print("translation_vector: ", translation_vector)
print("extrinsic_matrix: ", extrinsic_matrix)
print("camera_to_lidar: ", camera_to_lidar)

# # Save camera parameters to config.yaml
# config_data = {
#     'lens': 'pinhole',
#     'fx': camera_matrix[0, 0],
#     'fy': camera_matrix[1, 1],
#     'cx': camera_matrix[0, 2],
#     'cy': camera_matrix[1, 2],
#     'k1': dist[0, 0],
#     'k2': dist[0, 1],
#     'p1/k3': dist[0, 2],
#     'p2/k4': dist[0, 3]
# }

# with open('config.yaml', 'w') as f:
#     yaml.dump(config_data, f)

# # Save calibration data to data.txt
# with open('data.txt', 'w') as f:
#     if isinstance(objpoints, list):
#         for i in range(len(objpoints)):
#             for j in range(objpoints[i].shape[0]):
#                 line = f"{objpoints[i][j][0]} {objpoints[i][j][1]} {imgpoints[i][j][0]} {imgpoints[i][j][1]}\n"
#                 f.write(line)
#     else:
#         for j in range(objpoints.shape[0]):
#             line = f"{objpoints[j][0]} {objpoints[j][1]} {imgpoints[j][0]} {imgpoints[j][1]}\n"
#             f.write(line)

cap.release()

##########################################
# import cv2
# import numpy as np
# import yaml

# def rmse(objp, imgp, K, D, rvec, tvec):
#     predicted, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
#     predicted = predicted.squeeze()
#     imgp = imgp.squeeze()

#     pix_serr = []
#     for i in range(len(predicted)):
#         xp = predicted[i, 0]
#         yp = predicted[i, 1]
#         xo = imgp[i, 0]
#         yo = imgp[i, 1]
#         pix_serr.append((xp - xo) ** 2 + (yp - yo) ** 2)
#     ssum = sum(pix_serr)

#     return np.sqrt(ssum / len(pix_serr))

# def calibrate_camera(config_file, data_file, result_file):
#     with open(config_file, 'r') as f:
#         config = yaml.safe_load(f)
#         lens = config['lens']
#         fx = float(config['fx'])
#         fy = float(config['fy'])
#         cx = float(config['cx'])
#         cy = float(config['cy'])
#         k1 = float(config['k1'])
#         k2 = float(config['k2'])
#         p1 = float(config['p1/k3'])
#         p2 = float(config['p2/k4'])

#     K = np.array([[fx, 0.0, cx],
#                   [0.0, fy, cy],
#                   [0.0, 0.0, 1.0]])
#     D = np.array([k1, k2, p1, p2])
#     print("Camera parameters")
#     print("Lens =", lens)
#     print("K =")
#     print(K)
#     print("D =")
#     print(D)

#     imgp = []
#     objp = []
#     with open(data_file, 'r') as f:
#         for line in f:
#             data = line.split()
#             objp.append([float(data[0]), float(data[1]), 0.0])
#             imgp.append([float(data[2]), float(data[3])])

#     imgp = np.array(imgp, dtype=np.float32)
#     objp = np.array(objp, dtype=np.float32)

#     D_0 = np.array([0.0, 0.0, 0.0, 0.0])
#     retval, rvec, tvec = cv2.solvePnP(objp, imgp, K, D_0, flags=cv2.SOLVEPNP_ITERATIVE)
#     rmat, _ = cv2.Rodrigues(rvec)

#     print("Transform from camera to laser")
#     print("T = ")
#     print(tvec)
#     print("R = ")
#     print(rmat)

#     print("RMSE in pixel =", rmse(objp, imgp, K, D, rvec, tvec))

#     q = cv2.RQDecomp3x3(rmat)[0]
#     q = q.squeeze()

#     with open(result_file, 'w') as f:
#         f.write("%f %f %f %f %f %f %f" % (q[0], q[1], q[2], q[3], tvec[0], tvec[1], tvec[2]))

#     print("Result output format: qx qy qz qw tx ty tz")

# # Usage example
# calibrate_camera("config.yaml", "data.txt", "result.txt")