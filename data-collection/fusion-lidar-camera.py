import cv2
import numpy as np
import json
import yaml

def get_z(T_cam_world, T_world_pc, K):
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    proj_mat = np.dot(K, np.hstack((R, t[:, np.newaxis])))
    xyz_hom = np.hstack((T_world_pc, np.ones((T_world_pc.shape[0], 1))))
    xy_hom = np.dot(proj_mat, xyz_hom.T).T
    z = xy_hom[:, -1]
    z = np.asarray(z).squeeze()
    return z

def extract(point):
    return [point[0], point[1], point[2]]

def callback(image):
    img = image['data']

    # Load LiDAR data from lidarJson.json
    jsonLidarPath = "../lidarJson.json"  # Replace with the actual path to your LiDAR JSON file
    with open(jsonLidarPath) as f:
        try:
            lidar_data = json.load(f)
        except:
            lidar_data = {"data": [[0, 0, 0]]}

    # LiDAR data
    dataLidar = np.array(lidar_data['data'])
    lidar_angles = np.radians(dataLidar[:, 1])  # Convert angle to radians
    lidar_distances = dataLidar[:, 2]  # Lidar distance
    lidar_angles = 1.5*np.pi - lidar_angles

    # Project lidar data into camera frame
    lidar_points = np.zeros((len(lidar_distances), 3))
    for i in range(len(lidar_distances)):
        distance = lidar_distances[i] / 10
        angle = lidar_angles[i]

        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        z = 10.5 # lidar height 10.5 cm from ground

        lidar_points[i] = [x, y, z]

    Z = get_z(q, lidar_points, K)
    lidar_points = lidar_points[Z > 0]
    if lens == 'pinhole':
        img_points, _ = cv2.projectPoints(lidar_points, rvec, tvec, K, D)
    elif lens == 'fisheye':
        lidar_points = np.reshape(lidar_points, (1, lidar_points.shape[0], lidar_points.shape[1]))
        img_points, _ = cv2.fisheye.projectPoints(lidar_points, rvec, tvec, K, D)
    img_points = np.squeeze(img_points)
    for i in range(len(img_points)):
        try:
            cv2.circle(img, (int(round(img_points[i][0])), int(round(img_points[i][1]))), laser_point_radius, (0, 255, 0), 1)
        except OverflowError:
            continue
    cv2.imshow("Reprojection", img)
    cv2.waitKey(1)

# Replace with the actual calibration file path
calib_file = "result.txt"
# Replace with the actual config file path
config_file = "config.yaml"
# Replace with the desired laser point radius
laser_point_radius = 1

# Load camera-to-laser calibration parameters
with open(calib_file, 'r') as f:
    data = f.readline().split()
    qx, qy, qz, qw, tx, ty, tz = map(float, data)
q = np.array([[qw, -qz, qy, tx],
              [qz, qw, -qx, ty],
              [-qy, qx, qw, tz],
              [0, 0, 0, 1]])
print("Extrinsic parameter - camera to laser")
print(q)
tvec = q[:3, 3]
rot_mat = q[:3, :3]
rvec, _ = cv2.Rodrigues(rot_mat)

# Load camera parameters from config file
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    lens = config['lens']
    fx = float(config['fx'])
    fy = float(config['fy'])
    cx = float(config['cx'])
    cy = float(config['cy'])
    k1 = float(config['k1'])
    k2 = float(config['k2'])
    p1 = float(config['p1/k3'])
    p2 = float(config['p2/k4'])

K = np.array([[fx, 0.0, cx],
              [0.0, fy, cy],
              [0.0, 0.0, 1.0]])
D = np.array([k1, k2, p1, p2])
print("Camera parameters")
print("Lens =", lens)
print("K =")
print(K)
print("D =")
print(D)

# OpenCV VideoCapture for webcam
cap = cv2.VideoCapture(1)  # Replace with the appropriate webcam index if not the default

while True:
    ret, frame = cap.read()
    if not ret:
        break
    image = {
        'data': frame,
    }
    callback(image)

cap.release()
cv2.destroyAllWindows()