import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
from numpy import array

from filters.kalman_filter import KalmanFilter
from utils.UtilsFilters import get_positon_vectors, plot_robot_trajectory

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

def fusion_test(ps__sensor1_vector, ps_sensor1_sigma, direction="X direction", dt = 0.1, do_plot=True):

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = array([[1., dt], [0., 1.]])
    kf.H = array([[1., 0.]])
    kf.x = array([[0.], [0.]])
    kf.Q *= array([[(dt ** 3) / 3, (dt ** 2) / 2],
                   [(dt ** 2) / 2, dt]]) * 0.02
    kf.P *= 100
    kf.R[0, 0] = ps_sensor1_sigma

    random.seed(1123)
    xs, zs, nom = [], [], []
    number_of_measurment = ps__sensor1_vector.shape[0]
    for i in range(0, number_of_measurment):
        m1 = ps__sensor1_vector[i]
        z = array([[m1]])
        kf.predict()
        kf.update(z)

        xs.append(kf.x.T[0])
        zs.append(z.T[0])
        nom.append(i)

    xs = array(xs)
    zs = array(zs)

    if do_plot:
        plt.figure(1)
        ax = plt.subplot(111)
        ts = np.arange(0, time_interval*number_of_measurment, time_interval)
        ax.plot(ts, zs[:, 0], label='Pos Sensor: ' + direction)
        ax.plot(ts, xs[:, 0], label='Kalman Filter: '+ direction)
        ax.legend(loc=4)
        plt.xlabel('time (sec)')
        plt.ylabel('meters')
        plt.title("Trajectory of Robot Estimation X and Y Direction")
    return xs[:, 0], kf



sensor_number=2
if(sensor_number==1):
    x_position, kf_x = fusion_test(seonsor1_position_x_reading, seonsor1_position_x_2sigma,
                             direction="X Direction", dt=time_interval)
    y_position, kf_y = fusion_test(seonsor1_position_y_reading, seonsor1_position_y_2sigma,
                             direction="Y Direction", dt=time_interval)
    print("Final value of P: "+ str(kf_x.P) + " x direction")
    print("Final value of P: " + str(kf_y.P) + " y direction")
    print("Final value of K: " + str(kf_x.K) + " x direction")
    print("Final value of K: " + str(kf_y.K) + " y direction")
    plot_robot_trajectory(x_position, y_position, time_interval, label="Predicted Trajectory")
    plot_robot_trajectory(seonsor1_position_x_reading, seonsor1_position_y_reading, time_interval,
                          label="Original Trajectory")

else:
    x_position, kf_x = fusion_test(seonsor2_position_x_reading, seonsor2_position_x_2sigma,
                             direction="X Direction", dt=time_interval)
    y_position, kf_y = fusion_test(seonsor2_position_y_reading, seonsor2_position_y_2sigma,
                             direction="Y Direction", dt=time_interval)
    plot_robot_trajectory(x_position, y_position, time_interval, label="Predicted Trajectory")
    plot_robot_trajectory(seonsor2_position_x_reading, seonsor2_position_y_reading, time_interval,
                          label="Original Trajectory")
    print("Final value of P: " + str(kf_x.P) + " x direction")
    print("Final value of P: " + str(kf_y.P) + " y direction")
    print("Final value of K: " + str(kf_x.K) + " x direction")
    print("Final value of K: " + str(kf_y.K) + " y direction")

    plt.show()

