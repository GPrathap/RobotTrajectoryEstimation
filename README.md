# Robot Trajectory Prediction

####Note
If basic information is required for Kalman filter, Extended Kalman
filter, Unscented Kalman filter and Particle filter, please go 
through this([Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) 
which I followed in order to understand the basic concepts and 
help a lot to implement this task. 

####Installing prerequisites
Required libraries are listed in **requirement.txt**. Please install those libraries before using this.

####Problem Statement
Robot trajectory needs to predicted with the help of sensor fusion.
To do this there are few assumptions to be made. Lego robot motion 
model is constant velocity based model. So this model can model as a 
first order system. To predict the trajectory velocity and position of
the robot are considered.

####Problem Formulation
#####Design State Transition Function and Measurment Function
System has position and constant velocity, so the state variable needs both of these. The matrix
formulation could be

![](../master/images/formulation.png)

For data acquisition we used the AndroSensor android application which is freely available in
the google app store. This application is capable of measuring accelerometer readings(incl. linear
acceleration and gravity sensors), gyroscope readings, ambient magnetic field values, proximity
sensor readings and few other measurements. In this assignment, there were two mobile phone
were placed on robot and acquire separate sensor readings in parallel. Those two sensor reading
can be found in the sensor r eading directory. Here in order to measure the position of robot, linear
acceleration which is calculated using accelerometer and gravity sensors of the mobile phone is used.
Since sensor measures the linear acceleration of x, y and z direction, initially acceleration on each
direction need to be convert into position by tow-wise integration. Once position is derived from
the acceleration, it can be incorporate with the this model. Initially work with one accelerometer
and then use both sensors reading and do the sensor fusion to measure the robot position in x and
y direction separately. 


#####Design Process Noise and Measurement Noise
Since sensors are not perfect, some errors can be occurred during its measurement. Same goes for
processing the model as well. Because of some internal and external effects on configuration space.
In here[1], discrete time Wiener process is used to define the noise of the process. Thus, throughout
the this assignment, Q or processing noise is deducted using discrete time Wiener process.

To measure the noise of the sensors (R) there is no standard methods. What I have done
is, assume such that there is no correlation between two sensors and variance of each sensor is
measured by using collected sensor data. If the model is incorporate with two sensors.

#####Initial Conditions
Set the initial position at (0) with a velocity of (0). Covariance matrix P sets to a large value like
100.
Let’s consider when two sensors are being used,


![](../master/images/initcondition.png)

where ∆t = 0.1s and σ x 2 which is variance of sensor 1 and 
σ y 2 is the variance of sensor 2. Also
velocity is set to 20m/s of Lego robot.


Please read **TrajectoryEstimation.pdf** for details explanation. 


####Result Visualization 
1. See the result with two sensors, 
there is function called **plot_sensor_fusion_result()** 
just uncomment and run the required file. 
2. See the result of individual sensor for a given filer, 
change the **sensor_number** into 1 or 2. Then run the required file. 


