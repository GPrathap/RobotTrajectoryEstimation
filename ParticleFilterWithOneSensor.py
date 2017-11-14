import matplotlib.pyplot as plt
import numpy as np

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

def fusion_test(ps__sensor1_vector, ps_sensor1_sigma, robot_sigma,number_of_points, direction="X direction", dt = 0.1, do_plot=True):

    x = 0.1
    x_P = np.zeros(number_of_points)
    for i in range(0, number_of_points):
        x_P[i]= x + np.sqrt(robot_sigma) + np.random.uniform(-1, 1, 1)[0]

    number_of_measurment = ps__sensor1_vector.shape[0]
    z_out = np.zeros(number_of_measurment)
    x_out = np.zeros(number_of_measurment)
    x_P_update = np.zeros(number_of_points)
    x_est_out = np.zeros(number_of_measurment)
    z_update = np.zeros(number_of_points)
    P_w = np.zeros(number_of_points)

    for j in range(0, number_of_measurment):
        x = x + 0.1 * 20  # robot position estimation model
        z = ps__sensor1_vector[j]
        for k in range(0, number_of_points):
            x_P_update[k] = x_P[k] + 0.1 * 20
            z_update[k] = z + np.sqrt(np.power((x_P_update[k] - (z + np.sqrt(ps_sensor1_sigma))), 2))
            P_w[k] = ((1 / np.sqrt(2 * np.pi * ps_sensor1_sigma)) *
                      np.exp(-np.power(z - z_update[k], 2) / 2 * ps_sensor1_sigma))

        P_w = P_w / np.sum(P_w)
        cumsum = np.cumsum(P_w)

        for i in range(0, number_of_points):
            position = (np.random.rand() <= cumsum).sum()
            if (position == number_of_points):
                x_P[i] = x_P_update[position - 1]
            else:
                x_P[i] = x_P_update[position]

        x_est = np.mean(x_P)
        x_out[j]=x
        z_out[j]=z
        x_est_out[j]=x_est

    plt.figure(1)
    ax = plt.subplot(111)
    ts = np.arange(0, dt*number_of_measurment, dt)
    ax.plot(ts, x_out, label='Pos Sensor: '+ direction)
    ax.plot(ts, x_est_out, label='Kalman Filter: '+ direction)
    ax.legend(loc=4)
    plt.xlabel('time (sec)')
    plt.ylabel('meters')
    plt.title("Trajectory of Robot Estimation X and Y Direction")

    return x_est_out

sensor_number=2

if(sensor_number==1):
    x_position = fusion_test(seonsor1_position_x_reading, seonsor1_position_x_2sigma,
                             1, 100, direction="X Direction", dt=0.1)

    y_position = fusion_test(seonsor1_position_y_reading, seonsor1_position_y_2sigma,
                             1, 100, direction="Y Direction", dt=0.1)
    plot_robot_trajectory(x_position, y_position, time_interval, label="Predicted Trajectory")
    plot_robot_trajectory(seonsor1_position_x_reading, seonsor1_position_y_reading, time_interval,
                          label="Original Trajectory")
    plt.show()
else:
    x_position = fusion_test(seonsor2_position_x_reading, seonsor1_position_x_2sigma,
                             1, 100, direction="X Direction", dt=0.1)

    y_position = fusion_test(seonsor2_position_y_reading, seonsor1_position_y_2sigma,
                             1, 100, direction="Y Direction", dt=0.1)
    plot_robot_trajectory(x_position, y_position, time_interval, label="Predicted Trajectory")
    plot_robot_trajectory(seonsor2_position_x_reading, seonsor2_position_y_reading, time_interval,
                          label="Original Trajectory")
    plt.show()

