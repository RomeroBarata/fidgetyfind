import math

import numpy as np


def get_reference_length(skeleton: np.ndarray, reference_joints_indices):
    return np.linalg.norm(skeleton[reference_joints_indices[1], :] - skeleton[reference_joints_indices[0], :])


def get_angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    angle = np.arctan2(det, dot)
    return angle


def rotate(x, a, scale='deg'):
    if scale == 'deg':
        a = np.deg2rad(a)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.dot(rot, x)
