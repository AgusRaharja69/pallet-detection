from datetime import datetime
import json

from rplidar import RPLidar

PORT_NAME = 'COM14'
BAUDRATE: int = 115200
TIMEOUT: int = 1
DMAX: int = 4000
IMIN: int = 0
IMAX: int = 50
raw = True

def run():
    lidar = RPLidar(PORT_NAME, baudrate=BAUDRATE, timeout=3)
    try:
        for val in lidar.iter_scans(min_len=100, scan_type='normal', max_buf_meas=False):
            lidarData = {}
            lidarData["data"] = val
            with open('../lidarJson.json','w') as outfile :
                json.dump(lidarData, outfile, indent=2)
                lidarData["data"] = [0,0,0]
    except KeyboardInterrupt:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()

if __name__ == '__main__':
    run()