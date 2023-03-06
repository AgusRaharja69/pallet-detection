import json
import cv2
import numpy as np

# Connect to the webcam
cap = cv2.VideoCapture(1)
jsonLidarPath = '../lidarJson.json'

# Load the camera intrinsic parameters
################# camera_calli_dfov.py
# fx = 1087.63669
# cx = 348.947942
# fy = 1073.63075
# cy = 208.271104

# Dist = [[-0.323174294, 17.1891349, -0.00898601993, 0.00782426231, -170.883413]]
# # Load the camera extrinsic parameters
# Rmat = [[[-0.99156241,  0.01316134,  0.12896031],
#         [-0.00121852, -0.99573486,  0.09225296],
#         [ 0.12962445,  0.09131742,  0.9873493 ]], 
#         [[ 0.99748596, -0.01171715,  0.06988893],
#         [ 0.00496609,  0.99536902,  0.09599927],
#         [-0.07069011, -0.09541085,  0.99292481]], 
#         [[ 0.98425335, -0.01567026,  0.17606756],
#         [-0.00136873,  0.99535725,  0.09623966],
#         [-0.17675822, -0.0949652 ,  0.97966226]]]
# Tmat = [[[ 3.95183824],
#         [ 2.81445897],
#         [23.57023593]], 
#         [[-3.91996908],
#         [-1.63936852],
#         [19.30888547]], 
#         [[-2.22447691],
#         [-1.586544  ],
#         [19.54668881]]]

################ camera_calli_img.py // program 1
fx = 937.62318207
cx = 313.31072176
fy = 931.15422523
cy = 230.73339846

Dist = [[-0.00494782999, 2.55886434, -0.00426563426, -0.00540269681, -18.3530389]]
# Load the camera extrinsic parameters
Rvecs = [[[-0.20275292],
        [-0.12887969],
        [-3.12250216]], 
        [[-0.07783155],
        [ 0.0832318 ],
        [ 0.01183084]], 
        [[-0.08025628],
        [ 0.18444599],
        [ 0.0110054 ]], 
        [[-0.07892862],
        [ 0.02545437],
        [ 0.01241748]], 
        [[-0.07433848],
        [-0.08728151],
        [ 0.01386713]]]

Rmat = [[[-0.99156866,  0.015161  ,  0.12869244],
        [-0.00450407, -0.99656434,  0.08269964],
        [ 0.1295041 ,  0.08142273,  0.98823025]], 
        [[ 0.99647011, -0.01504046,  0.08258994],        
        [ 0.00856948,  0.99690453,  0.07815322],
        [-0.08350975, -0.07716959,  0.99351446]], 
        [[ 0.98298693, -0.01830759,  0.18276084],        
        [ 0.00355463,  0.99672999,  0.08072607],
        [-0.1836411 , -0.07870303,  0.97983763]], 
        [[ 0.99959918, -0.01340688,  0.02493479],        
        [ 0.01139898,  0.99680991,  0.0789941 ],
        [-0.02591431, -0.07867821,  0.99656319]], 
        [[ 0.99609916, -0.01059574, -0.08760249],        
        [ 0.01707691,  0.99714392,  0.07356885],
        [ 0.08657278, -0.07477785,  0.99343517]]]

Tmat = [[[ 308.92784198],[ 150.66904871],[1323.77678095]], 
        [[-213.79997389],[-133.05343448],[1090.58226142]], 
        [[-103.35163874],[-129.83539995],[1106.95485043]], 
        [[-270.70653691],[-134.17029586],[1078.5643076 ]], 
        [[-396.97799953],[-120.14831888],[1482.07043258]]]

# ################ camera_calli_img.py // program 2
# fx = 254.59472085
# cx = 343.78050744
# fy = 253.62067987
# cy = 268.66154398

# Dist = [[-0.00285454, 0.00841366, -0.00042065, -0.00220787, -0.00252511]]
# # Load the camera extrinsic parameters
# Rmat = [[[ 0.9982316 , -0.01661928,  0.05707428],
#         [ 0.01506291,  0.99950578,  0.02759188],
#         [-0.05750463, -0.02668338,  0.99798858]], 
#         [[ 0.99823157, -0.01661947,  0.05707476],        
#         [ 0.01506305,  0.99950575,  0.02759278],
#         [-0.05750513, -0.02668426,  0.99798853]], 
#         [[-0.9990243 ,  0.01137861,  0.04267284],        
#         [-0.01049889, -0.99972887,  0.02078341],
#         [ 0.04289775,  0.02031511,  0.9988729 ]], 
#         [[ 0.99961512, -0.01240202,  0.0248153 ],        
#         [ 0.01185208,  0.99968358,  0.02218692],
#         [-0.02508261, -0.02188427,  0.99944582]], 
#         [[ 0.99838239, -0.01325019,  0.05529048],        
#         [ 0.01227778,  0.99976457,  0.01789007],
#         [-0.05551451, -0.01718229,  0.99831003]], 
#         [[ 0.99990396, -0.011698  ,  0.00743087],        
#         [ 0.01153205,  0.99969154,  0.02199629],
#         [-0.00768589, -0.02190849,  0.99973044]], 
#         [[ 0.99974318, -0.01242331, -0.01895344],        
#         [ 0.01273383,  0.99978522,  0.01635137],
#         [ 0.01874623, -0.01658852,  0.99968665]]]

# Tmat = [[[-2.77426843],[-3.19343659],[ 5.07088027]], 
#         [[-2.77426942],[-3.19343548],[ 5.07088073]], 
#         [[4.1052927 ],[1.48937974],[5.52312688]], 
#         [[-3.83552888],[-2.72891779],[ 4.57088515]], 
#         [[-2.17061334],[-2.69646003],[ 4.65940084]], 
#         [[-4.69968487],[-2.73548477],[ 4.5167612 ]], 
#         [[-6.87529967],[-2.77255947],[ 6.23290888]]]


R = np.array(Rmat[1])
t = np.array(Tmat[1])
rVec = np.array(Rvecs[1])
# R = np.array([[ 1, 0, 0],
#                 [ 0, 1, 0 ],
#                 [ 0, 0, 1 ]])
# t = np.array([[ 0 ],
#                 [0],
#                 [0]])
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

while True:
    # Get a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the RPLidar scan to a point cloud
    with open(jsonLidarPath, 'r') as j:
        try :
            lidarData = json.loads(j.read())
            lidarAngle = [a[1] for a in lidarData['data']]
            lidarDist = [b[2] for b in lidarData['data']]
            # print("new")
        except :
            # print("old")
            lidarAngle, lidarDist = lidarAngle, lidarDist
    
    x = lidarDist * np.cos(lidarAngle)
    y = lidarDist * np.sin(lidarAngle)

    point_cloud = [[x[i], y[i], 0] for i in range(len(x))]

    # print(point_cloud)

    # # Convert the point cloud to a numpy array
    point_cloud = np.array(point_cloud)

    # # Transform the point cloud to the camera coordinate frame
    # point_cloud_cam = R.dot(point_cloud.T) + t

    # # Project the point cloud onto the image plane
    # point_cloud_image = K.dot(point_cloud_cam)
    # point_cloud_image /= point_cloud_image[2,:]
    # point_cloud_image = point_cloud_image[:2,:].T

    for point in point_cloud:
        # Calculate 2D pixel
        point_2d, _ = cv2.projectPoints(point, R, t, K, None)

        # Extract the X, Y pixel values
        xFrame, yFrame = point_2d[0][0][0], point_2d[0][0][1]
        cv2.circle(frame, (int(xFrame), int(yFrame)), 2, (255, 0, 0), -1)
        # print([xFrame,yFrame])

    # # Draw the point cloud on the image
    # for p in point_cloud_image:
    #     cv2.circle(frame, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)
    #     # cv2.circle(frame, (xFrame, yFrame), 2, (255, 0, 0), -1)

    # Display the image
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
