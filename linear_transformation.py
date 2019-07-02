import math
import numpy as np


def body_space(body_matrix):
    """
    :param body_matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    :return: two tuples for max and min values on x and y

    """
    #funziona solo se matrice ha x e y - qui frame,x,y,score
    body_matrix = body_matrix[:,:,[1,2]]
    max_a = np.nanmax(body_matrix, axis=1)
    max_xy = np.nanmax(max_a, axis=0)
    min_a = np.nanmin(body_matrix, axis=1)
    min_xy = np.nanmin(min_a, axis=0)
    return max_xy, min_xy


def scale_function(matrix, box_dim):
    """
    :param matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    :param box_dim: max distance between two points along x or y
    :return: matrix of landmarks in scaled space
    """
    new_dimension = 600

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not math.isnan(matrix[i][j][0]):
                matrix[i][j][1] = int((matrix[i][j][1] / box_dim) * new_dimension)
            if not math.isnan(matrix[i][j][1]):
                matrix[i][j][2] = int((matrix[i][j][2] / box_dim) * new_dimension)
    return matrix


def shift_function(matrix, max_xy, min_xy):
    """

    :param matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    :param max_xy: tupla of max_x and max_y
    :param min_xy: tupla of min_x and min_y
    :return: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    """

    dist_x = max_xy[0] - min_xy[0]
    dist_y = max_xy[1] - min_xy[1]
    if dist_x > dist_y:
        box_dim = dist_x
    else:
        box_dim = dist_y
    med_x = (max_xy[0] + min_xy[0]) // 2
    med_y = (max_xy[1] + min_xy[1]) // 2
    shift_x = med_x - box_dim // 2
    shift_y = med_y - box_dim // 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not math.isnan(matrix[i][j][0]):
                matrix[i][j][1] = matrix[i][j][1] - shift_x
            if not math.isnan(matrix[i][j][1]):
                matrix[i][j][2] = matrix[i][j][2] - shift_y
    return matrix, box_dim


def rotation_function(matrix, rot_ref):
    """

    :param matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    :param rot_ref: two indices of two rotation reference's landmarks
    :return: matrix with rotation's points
    """

    point_a = matrix[0][rot_ref[0]]
    point_b = matrix[0][rot_ref[1]]
    s_y = point_a[2] - point_b[2]
    ip = math.sqrt((point_a[1] - point_b[1]) ** 2 + (point_a[2] - point_b[2]) ** 2)
    angle = math.asin(s_y / ip)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            print(matrix[i][j])
            #if (not math.isnan(matrix[i][j][1])) and (not math.isnan(matrix[i][j][2])):
            if ( not matrix[i][j][1] is None) and ( not matrix[i][j][2] is None):
                matrix[i][j] = rotate_around_point_highperf(matrix[i][j], -angle, point_a)
            else:
                matrix[i][j][1] = matrix[i-1][j][1]
                matrix[i][j][2] = matrix[i-1][j - 15][2]
                matrix[i][j][3] = matrix[i-1][j][3]
    return matrix


def rotate_around_point_highperf(point, angle, origin):
    """

    :param point: point to rotate
    :param angle: rotation angle in radians
    :param origin: center of rotation
    :return: tuple of coordinate's point rotated
    """
    fr, x, y,score = point
    _, offset_x, offset_y, __ = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(angle)
    sin_rad = math.sin(angle)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return fr, qx, qy,score


def linear_transformation(matrix, rotation_ref=(1,5)):
    matrix = rotation_function(matrix,rotation_ref)
    maxT, minT = body_space(matrix)
    matrix, box_dim = shift_function(matrix, maxT, minT)
    matrix = scale_function(matrix, box_dim)
    return matrix