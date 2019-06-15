import csv
import cv2 as cv2
import time
import math
import numpy as np
import body_dictionary as body_dic

body = body_dic.body()


def csv_to_matrix(path, int_path):
    # input:
    #      path:path of file csv with landmarks
    #      int_path: path of file txt with interest point
    # output:
    #      b: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    a = []
    interest = []
    temp = open(int_path, 'r')
    for elem in temp:
        interest.append(elem)
    with open(path, 'r') as fin:
        reader = csv.reader(fin)
        for row in reader:
            a.append(row)
    a = np.asarray(a)
    a = np.reshape(a, (-1, 25, 3))
    b = np.zeros(shape=(a.shape[0], a.shape[1], 2))
    j = 0
    for element in a:
        for body_part in interest:
            part = body_part.replace('\n', '')
            i = int(body.dictionary[part])
            if element[i][1] != '' and element[i][2] != '':
                b[j][i][0] = int(element[i][1])
                b[j][i][1] = int(element[i][2])
            else:
                b[j][i][0] = None
                b[j][i][1] = None
        j = j + 1
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if b[i][j][0] == 0 and b[i][j][1] == 0:
                b[i][j][0] = None
                b[i][j][1] = None
    return b


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


def linear_transform(matrix, max_xy, min_xy):
    # input:
    #      matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    #      max : tupla of max_x and max_y
    #      min : tupla of min_x and min_y
    # output:
    #      matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    dist_x = max_xy[0] - min_xy[0]
    dist_y = max_xy[1] - min_xy[1]
    new_dimension = 600
    if dist_x > dist_y:
        old_dimension = dist_x
    else:
        old_dimension = dist_y
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not math.isnan(matrix[i][j][0]):
                print(matrix[i][j])
                matrix[i][j][0] = int((matrix[i][j][0]/old_dimension)*new_dimension)
            if not math.isnan(matrix[i][j][1]):
                matrix[i][j][1] = int((matrix[i][j][1]/old_dimension)*new_dimension)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # print(matrix[i][j])
            if not math.isnan(matrix[i][j][0]):
                matrix[i][j][0] = matrix[i][j][0] - min_xy[0]
            if not math.isnan(matrix[i][j][1]):
                matrix[i][j][1] = matrix[i][j][1] - min_xy[1]
    return matrix


def let_me_see(matrix):
    # input:
    # matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    connection = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                  (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]
    color = [(255, 0, 0), (251, 49, 229), (106, 49, 229), (255, 255, 0), (64, 255, 0),
             (0, 128, 255), (255, 128, 0), (128, 0, 255), (255, 0, 255), (255, 0, 128),
             (255, 0, 64), (0, 128, 255), (0, 230, 0), (128, 0, 255), (251, 49, 229),
             (255, 0, 0), (251, 49, 229), (106, 49, 229), (255, 255, 0), (64, 255, 0),
             (0, 128, 255), (255, 128, 0), (128, 0, 255), (255, 0, 255), (255, 0, 128)
             ]
    for i in range(matrix.shape[0]):
        bframe = np.zeros((650, 650, 3), np.uint8)
        overlay = bframe
        for j in range(matrix.shape[1]):
            if (not math.isnan(matrix[i][j][0])) and (not math.isnan(matrix[i][j][0])):
                x = int(matrix[i][j][0])
                y = int(matrix[i][j][1])
                cv2.circle(overlay, (x, y), 5, color[j], -1)
        for element in connection:
            if (not math.isnan(matrix[i][element[0]][0])) and (not math.isnan(matrix[i][element[1]][0])):
                point_a = (int(matrix[i][element[0]][0]), int(matrix[i][element[0]][1]))
                point_b = (int(matrix[i][element[1]][0]), int(matrix[i][element[1]][1]))
                cv2.line(bframe, point_a, point_b, color[element[0]],4)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, bframe, 1 - alpha,0, bframe)
        cv2.imshow('output', bframe)
        time.sleep(0.17)
        k = cv2.waitKey(1)
        if k == 27:
            break;


giovi_path = 'move/arms_warmup/giovi.csv'
bobo_path = 'move/arms_warmup/bobo.csv'
interest_path = 'move/arms_warmup/interest_point.txt'

giovi_matrix = csv_to_matrix(giovi_path, interest_path)
bobo_matrix = csv_to_matrix(bobo_path, interest_path)
maxG, minG = body_space(giovi_matrix)
maxB, minB = body_space(bobo_matrix)
poldo = linear_transform(bobo_matrix, maxB, minB)
let_me_see(poldo)
