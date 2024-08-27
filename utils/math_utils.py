from math import sqrt

import numpy as np
from math import exp


def check_division_by_0(value, epsilon=0.01):
    """
    Check for division by zero and adjust the value if necessary.

    Args:
        value (float): The value to check.
        epsilon (float): The minimum value to return.

    Returns:
        float: The adjusted value.
    """
    if value < epsilon:
        value = epsilon
    return value


def sanchez_matilla(box1, box2, w=1280, h=360):
    """
    Calculate the linear cost between two bounding boxes.

    Args:
        box1 (list): First bounding box.
        box2 (list): Second bounding box.
        w (int): Width of the image.
        h (int): Height of the image.

    Returns:
        float: The linear cost.
    """
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


def yu(box1, box2):
    """
    Calculate the exponential cost between two bounding boxes.

    Args:
        box1 (list): First bounding box.
        box2 (list): Second bounding box.

    Returns:
        float: The exponential cost.
    """
    w1 = 0.5
    w2 = 1.5
    a = (box1[0] - box2[0]) / check_division_by_0(box1[2])
    a_2 = pow(a, 2)
    b = (box1[1] - box2[1]) / check_division_by_0(box1[3])
    b_2 = pow(b, 2)
    ab = (a_2 + b_2) * w1 * (-1)
    c = abs(box1[3] - box2[3]) / (box1[3] + box2[3])
    d = abs(box1[2] - box2[2]) / (box1[2] + box2[2])
    cd = (c + d) * w2 * (-1)
    exponential_cost = exp(ab) * exp(cd)
    return exponential_cost


def cosine_similarity(a, b, data_is_normalized=False):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        a (numpy.ndarray): First vector.
        b (numpy.ndarray): Second vector.
        data_is_normalized (bool): If True, assume the data is already normalized.

    Returns:
        float: The cosine similarity.
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T)
