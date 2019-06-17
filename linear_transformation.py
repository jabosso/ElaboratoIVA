import math
import numpy as np


def body_space(body_matrix):
    # input:
    #      body_matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    # output:
    #      max, min: two tuples for max and min values on x and y
    max_a = np.nanmax(body_matrix, axis=1)
    max_xy = np.nanmax(max_a, axis=0)
    min_a = np.nanmin(body_matrix, axis=1)
    min_xy = np.nanmin(min_a, axis=0)
    return max_xy, min_xy


def scale_function(matrix, box_dim):

    new_dimension = 600

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not math.isnan(matrix[i][j][0]):
                matrix[i][j][0] = int((matrix[i][j][0]/box_dim)*new_dimension)
            if not math.isnan(matrix[i][j][1]):
                matrix[i][j][1] = int((matrix[i][j][1]/box_dim)*new_dimension)
    return matrix


def shift_function(matrix, max_xy, min_xy):
    # input:
    #      matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    #      max : tupla of max_x and max_y
    #      min : tupla of min_x and min_y
    # output:
    #      matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    dist_x = max_xy[0] - min_xy[0]
    dist_y = max_xy[1] - min_xy[1]
    if dist_x > dist_y:
        box_dim = dist_x
    else:
        box_dim = dist_y
    med_x = (max_xy[0]+min_xy[0])//2
    med_y = (max_xy[1] + min_xy[1]) // 2
    shift_x = med_x - box_dim//2
    shift_y = med_y - box_dim//2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not math.isnan(matrix[i][j][0]):
                matrix[i][j][0] = matrix[i][j][0] - shift_x
            if not math.isnan(matrix[i][j][1]):
                matrix[i][j][1] = matrix[i][j][1] - shift_y
    return matrix, box_dim


def rotation_function(matrix,rot_ref):
    point_a = matrix[0][rot_ref[0]]
    point_b = matrix[0][rot_ref[1]]

    s_y = point_a[1]- point_b[1]
    ip = math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
    angle = math.asin(s_y/ip)
    print (point_a, point_b, ip, angle)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (not math.isnan(matrix[i][j][0])) and (not math.isnan(matrix[i][j][0])):
                matrix[i][j]=rotate_around_point_highperf(matrix[i][j], -angle, point_a)
    return matrix


def rotate_around_point_highperf(xy, radians, origin):
    """Rotate a point around a given point.

    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy