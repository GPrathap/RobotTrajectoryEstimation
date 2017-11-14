import matplotlib.pyplot as plt
import numpy as np
from numpy import array

from filters.EKF import ExtendedKalmanFilter
from utils.UtilsFilters import get_positon_vectors
from utils.UtilsFilters import plot_robot_trajectory

time_interval = 0.1
position_vector1_x, position_vector1_y, position_vector1_z \
    = get_positon_vectors('./sensor_reading/Sensor_record_20171109_185848_AndroSensor1.csv', time_interval)

position_vector2_x, position_vector2_y, position_vector2_z \
    = get_positon_vectors('./sensor_reading/Sensor_record_20171109_185848_AndroSensor2.csv', time_interval)

seonsor1_position_x_reading = position_vector1_x[0][2:len(position_vector1_x[0])]
seonsor1_position_y_reading = position_vector1_y[0][2:len(position_vector1_y[0])]

seonsor1_position_x_2sigma = np.std(seonsor1_position_x_reading) * 2
seonsor1_position_y_2sigma = np.std(seonsor1_position_y_reading) * 2

seonsor2_position_x_reading = position_vector2_x[0][2:len(position_vector2_x[0])]
seonsor2_position_y_reading = position_vector2_y[0][2:len(position_vector2_y[0])]

seonsor2_position_x_2sigma = np.std(seonsor2_position_x_reading) * 2
seonsor2_position_y_2sigma = np.std(seonsor2_position_y_reading) * 2


def HJacobian_at(x):
    """ compute Jacobian of H matrix at x """
    H = np.array([[1., 0.], [1., 0.]])
    return H

def hx(x):
    """ compute measurement for slant range that would correspond to state x."""
    H = np.array([[1., 0.], [1., 0.]])
    return np.dot(H, np.transpose(x))


def fusion_test(ps__sensor1_vector, ps_sensor1_2sigma, ps__sensor2_vector, ps_sensor2_2sigma, direction,
                dt=time_interval, do_plot=True):

    rk = ExtendedKalmanFilter(dim_x=2, dim_z=2)
    rk.x = np.array([0., 0.0])
    rk.F = np.array([[1., dt], [0., 1.]])
    rk.R = np.diag([ps_sensor1_2sigma, ps_sensor2_2sigma])
    rk.Q = np.array([[(dt ** 3) / 3, (dt ** 2) / 2],
                     [(dt ** 2) / 2, dt]]) * 0.02
    rk.P *= 100
    xs, track = [], []
    zs = []
    nom = []

    number_of_measurment = ps__sensor1_vector.shape[0]
    for i in range(0, number_of_measurment):
        m1 = ps__sensor1_vector[i]
        m2 = ps__sensor2_vector[i]
        z = np.array([m1, m2])
        rk.update(z, HJacobian_at, hx)
        rk.predict()
        xs.append(rk.x)
        zs.append(z.copy())

    xs = array(xs)
    zs = array(zs)

    if do_plot:
        plt.figure(1)  # the first figure
        ax = plt.subplot(111)
        ts = np.arange(0, dt*number_of_measurment, dt)
        plt.plot(ts, zs[:, 0], label='Pos Sensor 01: ' + direction)
        plt.plot(ts, xs[:, 0], label='Kalman Filter:' + direction)
        plt.plot(ts, zs[:, 1], label='Pos Sensor 02: ' + direction)
        ax.legend(loc=8)
        plt.xlabel('time (sec)')
        plt.ylabel('meters')
        plt.title("Trajectory of Robot Estimation X and Y Direction")
    return xs[:, 0], rk



def plot_sensor_fusion_predicted_result():
    x_position, kf_x = fusion_test(seonsor1_position_x_reading, seonsor1_position_x_2sigma,
                             seonsor2_position_x_reading, seonsor2_position_x_2sigma,
                             direction="X Direction", dt=time_interval, do_plot=False)
    y_position, kf_y = fusion_test(seonsor1_position_y_reading, seonsor1_position_y_2sigma,
                             seonsor2_position_y_reading, seonsor2_position_y_2sigma,
                             direction="Y Direction", dt=time_interval, do_plot=False)
    plot_robot_trajectory(x_position, y_position, time_interval,label="Predicted Trajectory for Extended Kalman Filter")

def plot_sensor_fusion_result():
    x_position, kf_x = fusion_test(seonsor1_position_x_reading, seonsor1_position_x_2sigma,
                             seonsor2_position_x_reading, seonsor2_position_x_2sigma,
                             direction="X Direction", dt=time_interval)
    y_position, kf_y = fusion_test(seonsor1_position_y_reading, seonsor1_position_y_2sigma,
                             seonsor2_position_y_reading, seonsor2_position_y_2sigma,
                             direction="Y Direction", dt=time_interval)
    plot_robot_trajectory(x_position, y_position, time_interval,label="Predicted Trajectory")
    plot_robot_trajectory(seonsor1_position_x_reading, seonsor1_position_y_reading, time_interval,
                          label="Original Trajectory form Sensor 01")
    plot_robot_trajectory(seonsor2_position_x_reading, seonsor2_position_y_reading, time_interval,
                          label="Original Trajectory form Sensor 02")
    print("Final value of P: " + str(kf_x.P) + " x direction")
    print("Final value of P: " + str(kf_y.P) + " y direction")
    print("Final value of K: " + str(kf_x.K) + " x direction")
    print("Final value of K: " + str(kf_y.K) + " y direction")


# plot_sensor_fusion_result()
# plt.show()
