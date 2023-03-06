from rplidar import RPLidar

PORT_NAME = 'COM14'
BAUDRATE: int = 115200
TIMEOUT: int = 1
DMAX: int = 4000
IMIN: int = 0
IMAX: int = 50

class PrintColor:
    YELLOW = '\033[1;33;48m'
    BLUE = '\033[1;34;48m'
    PURPLE = '\033[1;35;48m'
    END = '\033[1;37;0m'

def find_zero_front(angle, distance):
    min_range = range(0, 10)
    max_range = range(350, 360)

    # print(angle)

    if int(angle) == 0:
        print(PrintColor.YELLOW + "angle: 0 distance: {} millimeter".format(distance) + PrintColor.END)

    if int(angle) in min_range:
        print(PrintColor.BLUE + "angle: {:.2f} distance: {} millimeter".format(angle, distance) + PrintColor.END)

    if int(angle) in max_range:
        print(PrintColor.PURPLE + "angle: {:.2f} distance: {} millimeter".format(angle, distance) + PrintColor.END)


def run():
    lidar = RPLidar(PORT_NAME, baudrate=BAUDRATE, timeout=TIMEOUT)
    try:
        for val in lidar.iter_measures(max_buf_meas=False):
            if val[3] != 0:
                print(val)
                find_zero_front(val[2], val[3])
    except KeyboardInterrupt:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()

if __name__ == '__main__':
    run()