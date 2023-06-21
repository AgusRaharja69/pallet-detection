import cv2
import numpy as np
import json

jsonLidarPath = '../lidarJson.json'


# with open(jsonLidarPath) as f:
#         try :
#             lidar_data = json.load(f)
#         except:
#             lidar_data = {"data": [[0,0,0]]}

#     # LiDAR data
#     dataLidar = np.array(lidar_data['data'])
#     angleRawLidar = np.radians(dataLidar[:, 1]) # Convert angle to radians
#     rangeLidar = dataLidar[:, 2]
#     angleLidar = np.pi - angleRawLidar #mirror