import matplotlib.pyplot as plt
import numpy as np
from numpy import array

from filters.EKF import ExtendedKalmanFilter
from utils.UtilsFilters import get_positon_vectors
from utils.UtilsFilters import plot_robot_trajectory

time_interval = 0.1
x, y, z \
    = get_positon_vectors('/home/geesara/sensorlog_accel_20200407_205230.csv', time_interval)

pose_data = np.array([np.reshape(x,(x.shape[1])),np.reshape(x,(x.shape[1])),np.reshape(x,(x.shape[1]))])

print(pose_data.shape)
np.savetxt("foo.csv", pose_data.transpose(), delimiter=",")

