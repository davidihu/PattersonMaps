# author: David Hurwitz
# started: 3/16/18
#

import numpy as np
from numpy import float32

# (matrix(3x3)) x (vector(3)) gives a rotated vector
class ARotationMatrix:

    # when constructing the matrix, Axis implicitly is a unit vector starting at the origin
    # Radians is the angle rotated about the Axis
    def __init__(self, Axis, Radians):

        # this is the rotation matrix
        self.m_matrix = np.zeros(shape=[3,3], dtype=float32)

        # make sure Axis is a unit vector
        Axis = Axis / np.linalg.norm(Axis)

        Row1 = np.array([1.0, 0.0, 0.0], dtype=float32)
        Row2 = np.array([0.0, 1.0, 0.0], dtype=float32)
        Row3 = np.array([0.0, 0.0, 1.0], dtype=float32)

        Row4 = np.array([Axis[0]*Axis[0], Axis[0]*Axis[1], Axis[0]*Axis[2]], dtype=float32)
        Row5 = np.array([Axis[1]*Axis[0], Axis[1]*Axis[1], Axis[1]*Axis[2]], dtype=float32)
        Row6 = np.array([Axis[2]*Axis[0], Axis[2]*Axis[1], Axis[2]*Axis[2]], dtype=float32)

        Row7 = np.array([     0.0,  Axis[2], -Axis[1]], dtype=float32)
        Row8 = np.array([-Axis[2],      0.0,  Axis[0]], dtype=float32)
        Row9 = np.array([ Axis[1], -Axis[0],      0.0], dtype=float32)

        Mat1 = np.array([Row1, Row2, Row3])
        Mat2 = np.array([Row4, Row5, Row6])
        Mat3 = np.array([Row7, Row8, Row9])

        self.m_matrix = Mat1*np.cos(Radians) + Mat2*(1-np.cos(Radians)) + Mat3*np.sin(Radians)

