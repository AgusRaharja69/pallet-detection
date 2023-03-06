import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import cv2

jsonLidarPath = '../lidarJson.json'
jsonLidarPathClean = '../lidarJsonClean.json'
lidarAngle, lidarDist, lidarQ = [],[],[]

def update(frame):
    with open(jsonLidarPath, 'r') as j:
        try :
            lidarData = json.loads(j.read())
            lidarQ = [q[0] for q in lidarData['data']]
            lidarAngle = [a[1] for a in lidarData['data']]
            lidarDist = [b[2] for b in lidarData['data']]
        except :
            lidarAngle, lidarDist, lidarQ = lidarAngle, lidarDist, lidarQ

    # x = [lidarDist[i]*np.cos(np.radians(lidarAngle[i])) for i in range(len(lidarAngle)) if lidarQ[i] == 15 ]
    # y = [lidarDist[i]*np.sin(np.radians(lidarAngle[i])) for i in range(len(lidarAngle)) if lidarQ[i] == 15 ]

    angles = [np.radians(lidarAngle[i]) for i in range(len(lidarAngle)) if lidarQ[i] == 15]
    ranges = [lidarDist[i] for i in range(len(lidarDist)) if lidarQ[i] == 15]
    
    # Clear previous plot and plot new data
    ax.clear()
    ax.scatter(angles, ranges, s=1, c='b')
    ax.set_xlim([-1000000, 1000000])
    ax.set_ylim([-1000000, 1000000])

# # Create figure and axes
# fig, ax = plt.subplots(figsize=(7, 7))

# # Create animation
# ani = animation.FuncAnimation(fig, update, interval=50)

# # Show plot
# plt.show()

fig = plt.figure()
ax = plt.subplot(111, projection='polar')

ax.set_rmax(4000)
ax.grid(True)
ani = animation.FuncAnimation(fig, update, interval=50)
plt.show()


# while(True):
#     with open(jsonLidarPath, 'r') as j:
#         try :
#             lidarData = json.loads(j.read())
#             lidarQ = [q[0] for q in lidarData['data']]
#             lidarAngle = [a[1] for a in lidarData['data']]
#             lidarDist = [b[2] for b in lidarData['data']]
#         except :
#             lidarAngle, lidarDist, lidarQ = lidarAngle, lidarDist, lidarQ

#     x = [lidarDist[i]*np.cos(np.radians(lidarAngle[i])) for i in range(len(lidarAngle)) if lidarQ[i] == 15 ]
#     y = [lidarDist[i]*np.sin(np.radians(lidarAngle[i])) for i in range(len(lidarAngle)) if lidarQ[i] == 15 ]

#     angles = [np.radians(lidarAngle[i]) for i in range(len(lidarAngle)) if lidarQ[i] == 15]
#     ranges = [lidarDist[i] for i in range(len(lidarDist)) if lidarQ[i] == 15]
    
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='polar')
#     ax.scatter(x, y, s=1, c='b')

#     # plt.clf()
#     # plt.scatter(angles, ranges)
#     # plt.pause(.1)
#     plt.show()

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break