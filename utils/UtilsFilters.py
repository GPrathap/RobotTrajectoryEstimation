import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, asarray, isscalar, eye, dot


def get_positon_vectors(csv_file_name, time_between_samples):
    df = pd.read_csv(csv_file_name)

    def _get_position(accelerations, time_between_samples):
        velocity = np.zeros([1, len(accelerations) + 1])
        velocity[0][0] = 0
        possition = np.zeros([1, velocity.shape[1] + 1])
        possition[0][0] = 0
        for i in range(0, len(accelerations)):
            velocity[0][i + 1] = accelerations[i] * time_between_samples + velocity[0][i]
        for i in range(0, velocity.shape[1]):
            possition[0][i + 1] = velocity[0][i] * time_between_samples + possition[0][i]
        return possition

    accelerations_init_x = df.iloc[:, 6:7].values.astype(np.float)
    accelerations_init_y = df.iloc[:, 7:8].values.astype(np.float)
    accelerations_init_z = df.iloc[:, 8:9].values.astype(np.float)

    accelerations_x = []
    accelerations_y = []
    accelerations_z = []

    for h in range(0, accelerations_init_x.shape[0]):
        accelerations_x.append(accelerations_init_x[h][0])
        accelerations_y.append(accelerations_init_y[h][0])
        accelerations_z.append(accelerations_init_z[h][0])

    position_vector_x = _get_position(accelerations_x, time_between_samples)
    position_vector_y = _get_position(accelerations_y, time_between_samples)
    position_vector_z = _get_position(accelerations_z, time_between_samples)

    return position_vector_x, position_vector_y, position_vector_z

def plot_robot_trajectory(x_position, y_position, time_interval, fignum=2, label=""):

    trasnformation_matrix = np.zeros([3,3])
    trasnformation_matrix[0,0]=1
    trasnformation_matrix[1, 1]=1
    trasnformation_matrix[2,2]=1
    trasnformation_matrix[0,2]= np.abs(x_position.min())
    trasnformation_matrix[1,2]= np.abs(y_position.min())

    compacted_position = np.ones([3, x_position.shape[0]])
    compacted_position[0]= x_position
    compacted_position[1]= y_position

    transformed_coordinate_position = np.dot(trasnformation_matrix, compacted_position)

    new_x_position = transformed_coordinate_position[0]
    new_y_position = transformed_coordinate_position[1]


    robot_position = np.sqrt(new_x_position*2 + new_y_position*2)
    ts = np.arange(0, time_interval*robot_position.shape[0], .1)
    plt.figure(2)
    ax = plt.subplot(111)
    ax.plot(ts, robot_position, label=label)
    ax.legend(loc=4)
    plt.title('Robot Trajectory')

def dot3(A,B,C):
    return dot(A, dot(B,C))

def dot4(A,B,C,D):
    return dot(A, dot(B, dot(C,D)))

def setter(value, dim_x, dim_y):
    v = array(value, dtype=float)
    if v.shape != (dim_x, dim_y):
        raise Exception('must have shape ({},{})'.format(dim_x, dim_y))
    return v

def setter_1d(value, dim_x):
    v = array(value, dtype=float)
    shape = v.shape
    if shape[0] != (dim_x) or v.ndim > 2 or (v.ndim==2 and shape[1] != 1):
        raise Exception('has shape {}, must have shape ({},{})'.format(shape, dim_x, 1))
    return v


def setter_scalar(value, dim_x):
    if isscalar(value):
        v = eye(dim_x) * value
    else:
        v = array(value, dtype=float)
        dim_x = v.shape[0]

    if v.shape != (dim_x, dim_x):
        raise Exception('must have shape ({},{})'.format(dim_x, dim_x))
    return v

def unscented_transform(sigmas, Wm, Wc, noise_cov=None,
                        mean_fn=None, residual_fn=None):
    kmax, n = sigmas.shape
    if mean_fn is None:
        # new mean is just the sum of the sigmas * weight
        x = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])
    else:
        x = mean_fn(sigmas, Wm)

    if residual_fn is None:
        y = sigmas - x[np.newaxis,:]
        P = y.T.dot(np.diag(Wc)).dot(y)
    else:
        P = np.zeros((n, n))
        for k in range(kmax):
            y = residual_fn(sigmas[k], x)
            P += Wc[k] * np.outer(y, y)

    if noise_cov is not None:
        P += noise_cov
    return (x, P)

