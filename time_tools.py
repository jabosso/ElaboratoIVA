import math
import numpy as np
from scipy.spatial.distance import cdist


def choosen_point(M):
    """
    Goal: determinate the point to max variance

    :param M: dataframe
    :return: point to max variance
    """
    dist = np.zeros((M.shape[1]))
    for i in range(M.shape[1]):
        dist_b = np.zeros((M.shape[0]))
        for j in range(M.shape[0]):
            if not math.isnan(M[j][i][0]) and not math.isnan(M[j][i][1]):
                temp = np.zeros((M.shape[0]))
                for k in range(M.shape[0]):
                    if not math.isnan(M[k][i][0]) and not math.isnan(M[k][i][1]):
                        temp[k] = math.sqrt((M[j][i][0] - M[k][i][0]) ** 2 + (M[j][i][1] - M[k][i][1]) ** 2)
                    else:
                        temp[k] = 0
                dist_b[j] = np.max(temp)
        dist[i] = np.max(dist_b)
    point = np.argmax(dist)
    return point


def cycle_identify(matrix):
    """
    Goal:

    :param matrix:
    :return:
    """
    p = choosen_point(matrix)
    M = matrix[:, p]
    c_M = cdist(M, M)
    max_ = np.max(c_M[0])
    min_ = np.min(c_M[0])
    threshold = (max_ + min_) / 3
    temp = []
    temp.append(0)
    for i in range(c_M.shape[1] - 1):
        if ((c_M[0][i] - threshold) * (c_M[0][i + 1] - threshold)) < 0:
            temp.append(i)
    temp.append(c_M.shape[1])
    midpoints = []
    for i in range(0, len(temp), 2):
        midpoints.append((temp[i + 1] + temp[i]) / 2)
    return midpoints

def generate_cycle_model(matrix, mid_points):
    """
    Goal:generate one dataframe for any cycle

    :param matrix:
    :param mid_points:
    :return
    """
    spl_mat = []
    for i in range(len(mid_points) - 1):
        start = int(mid_points[i])
        end = int(mid_points[i + 1])
        matr = matrix[start:end]
        spl_mat.append(matr)
    return spl_mat