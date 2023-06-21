import numpy as np
import cv2 as cv
import glob

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
chessboardSize = (7,5)
frameSize = (640,480)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 65
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calli_dfov/*.jpg')

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()

############## CALIBRATION #######################################################
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

############## UNDISTORTION #####################################################
img = cv.imread('calli_dfov/saved_img_5.jpg')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calli_dfov/result_saved_img_5.jpg', dst)

# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calli_dfov/result_saved_img_5_2.jpg', dst)

# Reprojection Error
mean_error = 0
# print("obj ", objpoints)
for i in range(len(objpoints)):
    # print("1 1 obj : ",objpoints[i])
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

# Extract the rotation and translation matrices from the estimated parameters
rotation_matrices = []
for rotation_vector in rvecs:
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    rotation_matrices.append(rotation_matrix)
# Rt_matirx = np.concatenate((rotation_matrices, tvecs), axis=1)

# Displaying required output
print(" Camera matrix:")
print(cameraMatrix)

print("\n Distortion coefficient:")
print(dist)

print("\n Rotation Vector:")
print(rvecs)

print("\n Rotation Matrix:")
print(rotation_matrices)

print("\n Translation Vectors:")
print(tvecs)

# print("\n Rt Matrix:")
# print(Rt_matirx)



# # # #############################################################
# # # # Import required modules
# import cv2
# import numpy as np
# import os
# import glob


# # Define the dimensions of checkerboard
# CHECKERBOARD = (7, 5)


# # stop the iteration when specified
# # accuracy, epsilon, is reached or
# # specified number of iterations are completed.
# criteria = (cv2.TERM_CRITERIA_EPS +
#             cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# # Vector for 3D points
# threedpoints = []

# # Vector for 2D points
# twodpoints = []


# #  3D points real world coordinates
# objectp3d = np.zeros((1, CHECKERBOARD[0]
#                       * CHECKERBOARD[1],
#                       3), np.float32)
# objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
#                                0:CHECKERBOARD[1]].T.reshape(-1, 2)
# prev_img_shape = None


# # Extracting path of individual image stored
# # in a given directory. Since no path is
# # specified, it will take current directory
# # jpg files alone
# images = glob.glob('*.jpg')

# for filename in images:
#     image = cv2.imread(filename)
#     grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Find the chess board corners
#     # If desired number of corners are
#     # found in the image then ret = true
#     ret, corners = cv2.findChessboardCorners(
#                     grayColor, CHECKERBOARD,
#                     cv2.CALIB_CB_ADAPTIVE_THRESH
#                     + cv2.CALIB_CB_FAST_CHECK +
#                     cv2.CALIB_CB_NORMALIZE_IMAGE)

#     # If desired number of corners can be detected then,
#     # refine the pixel coordinates and display
#     # them on the images of checker board
#     if ret == True:
#         threedpoints.append(objectp3d)

#         # Refining pixel coordinates
#         # for given 2d points.
#         corners2 = cv2.cornerSubPix(
#             grayColor, corners, (11, 11), (-1, -1), criteria)

#         twodpoints.append(corners2)

#         # Draw and display the corners
#         image = cv2.drawChessboardCorners(image,
#                                           CHECKERBOARD,
#                                           corners2, ret)

#     cv2.imwrite(filename=f"cali_{filename}.jpg", img=image)
#     cv2.imshow('img', image)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()

# h, w = image.shape[:2]


# # Perform camera calibration by
# # passing the value of above found out 3D points (threedpoints)
# # and its corresponding pixel coordinates of the
# # detected corners (twodpoints)
# ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
#     threedpoints, twodpoints, grayColor.shape[::-1], None, None)

# # Extract the rotation and translation matrices from the estimated parameters
# rotation_matrices = []
# for rotation_vector in r_vecs:
#     rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
#     rotation_matrices.append(rotation_matrix)
# translation_matrices = t_vecs

# # Displaying required output
# print(" Camera matrix:")
# print(matrix)

# print("\n Distortion coefficient:")
# print(distortion)

# print("\n Rotation Vectors:")
# print(rotation_matrices)

# print("\n Translation Vectors:")
# print(translation_matrices)