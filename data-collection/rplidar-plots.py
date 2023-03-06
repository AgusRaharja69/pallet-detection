import rplidar
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
PORT_NAME = 'COM14'
BAUDRATE: int = 115200
TIMEOUT: int = 1
DMAX: int = 4000
IMIN: int = 0
IMAX: int = 50

def update_line(num, iterator, line):
    scan = next(iterator)

    offsets = np.array([(np.radians(abs((meas[1]-180))), meas[2]) for meas in scan])

    # print("scan",scan)
    line.set_offsets(offsets)
    intens = np.array([meas[0] for meas in scan])
    # print("inten",intens)
    line.set_array(intens)
    # print("line", line)
    return line,

def run():
    lidar = rplidar.RPLidar(PORT_NAME, baudrate=BAUDRATE, timeout=TIMEOUT)
    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')

    line = ax.scatter([0, 0], [0, 0], s=5, c=[IMIN, IMAX],
                           cmap=plt.cm.Greys_r, lw=0)
    ax.set_rmax(DMAX)
    ax.grid(True)
    lidar.motor_speed = rplidar.MAX_MOTOR_PWM
    iterator = lidar.iter_scans(min_len=100, scan_type='normal', max_buf_meas=3000)
    ani = animation.FuncAnimation(fig, update_line,
        fargs=(iterator, line), interval=50)
    plt.show()
    lidar.stop()
    lidar.disconnect()
    
if __name__ == '__main__':
    run()