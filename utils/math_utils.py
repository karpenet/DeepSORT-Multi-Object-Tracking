from math import sqrt

import numpy as np


def check_division_by_0(value, epsilon=0.01):
    if value < epsilon:
        value = epsilon
    return value

def sanchez_matilla(box1, box2, w=1280, h=360):
    Q_dist = sqrt(pow(w, 2) + pow(h, 2))
    Q_shape = w * h
    distance_term = Q_dist / check_division_by_0(
        sqrt(pow(box1[0] - box2[0], 2) + pow(box1[1] - box2[1], 2))
    )
    shape_term = Q_shape / check_division_by_0(
        sqrt(pow(box1[2] - box2[2], 2) + pow(box1[3] - box2[3], 2))
    )
    linear_cost = distance_term * shape_term
    return linear_cost

def cosine_similarity(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T)