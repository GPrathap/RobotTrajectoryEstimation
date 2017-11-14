# -*- coding: utf-8 -*-
"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

import numpy as np
import scipy.linalg as linalg
from numpy import dot, zeros, eye

from utils.UtilsFilters import dot3, setter_1d, setter, setter_scalar


class ExtendedKalmanFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self._x = zeros((dim_x,1)) # state
        self._P = eye(dim_x)       # uncertainty covariance
        self._B = 0                # control transition matrix
        self._F = 0                # state transition matrix
        self._R = eye(dim_z)       # state uncertainty
        self._Q = eye(dim_x)       # process uncertainty
        self._y = zeros((dim_z, 1))
        self._I = np.eye(dim_x)
        self._K = []


    def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        F = self._F
        B = self._B
        P = self._P
        Q = self._Q
        R = self._R
        x = self._x


        H = HJacobian(x, *args)

        # predict step
        x = dot(F, x) + dot(B, u)
        P = dot3(F, P, F.T) + Q

        # update step
        S = dot3(H, P, H.T) + R
        K = dot3(P, H.T, linalg.inv (S))
        self._K = K

        self._x = x + dot(K, (z - Hx(x, *hx_args)))

        I_KH = self._I - dot(K, H)
        self._P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(), residual=np.subtract):

        P = self._P
        if R is None:
            R = self._R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        x = self._x

        H = HJacobian(x, *args)

        S = dot3(H, P, H.T) + R
        K = dot3(P, H.T, linalg.inv (S))
        self._K = K

        hx =  Hx(x, *hx_args)
        y = residual(z, hx)
        self._x = x + dot(K, y)

        I_KH = self._I - dot(K, H)
        self._P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    def predict_x(self, u=0):
        self._x = dot(self._F, self._x) + dot(self._B, u)


    def predict(self, u=0):
        self.predict_x(u)
        self._P = dot3(self._F, self._P, self._F.T) + self._Q

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = setter_scalar(value, self.dim_x)

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value):
        self._P = setter_scalar(value, self.dim_x)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        self._R = setter_scalar(value, self.dim_z)

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        self._F = setter(value, self.dim_x, self.dim_x)

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, value):
        self._B = setter(value, self.dim_x, self.dim_u)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = setter_1d(value, self.dim_x)

    @property
    def K(self):
        return self._K

    @property
    def y(self):
        return self._y

    @property
    def S(self):
        return self._S

