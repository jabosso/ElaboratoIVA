import csv
import cv2 as cv2
import time
import math
import numpy as np
import body_dictionary as body_dic
import matplotlib.pyplot as plt

body = body_dic.body()

connection = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
              (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]
color = [(255, 0, 0), (251, 49, 229), (106, 49, 229), (255, 255, 0), (64, 255, 0),
         (0, 128, 255), (255, 128, 0), (128, 0, 255), (255, 0, 255), (255, 0, 128),
         (255, 0, 64), (0, 128, 255), (0, 230, 0), (128, 0, 255), (251, 49, 229),
         (255, 0, 0), (251, 49, 229), (106, 49, 229), (255, 255, 0), (64, 255, 0),
         (0, 128, 255), (255, 128, 0), (128, 0, 255), (255, 0, 255), (255, 0, 128)
         ]


def csv_to_matrix(path, int_path):
    """

    :param path: path of file csv with landmarks
    :param int_path: path of file txt with interest point
    :return:  matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    """

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
            if element[i][0] != '' and element[i][1] != '':
                b[j][i][0] = int(element[i][0].replace("'",''))
                b[j][i][1] = int(element[i][1])
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

def funzion(m,int_path):
    interest = []
    temp = open(int_path, 'r')
    for elem in temp:
        interest.append(elem)
    b = np.zeros(shape=(m.shape[0], m.shape[1], 2))
    j = 0
    for element in m:
        for body_part in interest:
            part = body_part.replace('\n', '')
            print(part)
            i = int(body.dictionary[part])
            if element[i][0] != '' and element[i][1] != '':
                b[j][i][0] = int(element[i][0])
                b[j][i][1] = int(element[i][1])
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



def array_to_csv(v, name_file):
    with open('move/models/'+ name_file + '.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_NONE)
        employee_writer.writerow(v)



def matrix_to_csv(matrix, name_file):
    """
    :param matrix: matrix to convert in file csv
    :param name_file: part of name file csv
    """
    with open('move/models/'+ name_file +'/model.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_NONE)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                employee_writer.writerow(int(matrix[i][j]))


'''
def matrix_to_csv_3D(matrix, name_file):
    """
    :param matrix: matrix to convert in file csv
    :param name_file: part of name file csv
    """
    with open('matrix_to_csv_' + name_file + '.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_NONE)
        for i in range(matrix.shape[0]):
            employee_writer.writerow(matrix[i][:][0])

'''


def transform_matrix_noNan(matrix, cod):
    """

    :param matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    :param cod: 0 or 1
    :return: matrix with element no nan, replaced with 700 or -700
    """
    if cod == 0:
        value_ = 700
    else:
        value_ = -700
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if math.isnan(matrix[i][j][0]):
                matrix[i][j][0] = value_
                matrix[i][j][1] = value_
    return matrix


def compare_two_movements(matrix1, matrix2):
    """

    :param matrix1: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    :param matrix2: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    :return two matrix with element no nan
    """
    matrix1 = transform_matrix_noNan(matrix1, 0)
    matrix2 = transform_matrix_noNan(matrix2, 0)
    return matrix1, matrix2


def let_me_see(matrix):
    """
    Goal: show the movement of the person

    :param matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    """

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
                cv2.line(bframe, point_a, point_b, color[element[0]], 4)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, bframe, 1 - alpha, 0, bframe)
        cv2.imshow('output', bframe)
        time.sleep(0.17)
        k = cv2.waitKey(1)
        if k == 27:
            break;


def let_me_see_two_movements(matrix1, matrix2):
    """
    Goal:show the movement of two person simultaneously

    :param matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    """

    for i in range(matrix1.shape[0]):
        bframe = np.zeros((650, 650, 3), np.uint8)
        overlay = bframe
        for j in range(matrix1.shape[1]):
            if (not math.isnan(matrix1[i][j][0])) and (not math.isnan(matrix1[i][j][0])):
                x = int(matrix1[i][j][0])
                y = int(matrix1[i][j][1])
                cv2.circle(overlay, (x, y), 5, color[j], -1)
        for element in connection:
            if (not math.isnan(matrix1[i][element[0]][0])) and (not math.isnan(matrix1[i][element[1]][0])):
                point_a = (int(matrix1[i][element[0]][0]), int(matrix1[i][element[0]][1]))
                point_b = (int(matrix1[i][element[1]][0]), int(matrix1[i][element[1]][1]))
                cv2.line(overlay, point_a, point_b, color[1], 4)

        for j in range(matrix2.shape[1]):
            if (not math.isnan(matrix2[i][j][0])) and (not math.isnan(matrix2[i][j][0])):
                x = int(matrix2[i][j][0])
                y = int(matrix2[i][j][1])
                cv2.circle(bframe, (x, y), 5, color[j], -1)
        for element in connection:
            if (not math.isnan(matrix2[i][element[0]][0])) and (not math.isnan(matrix2[i][element[1]][0])):
                point_a = (int(matrix2[i][element[0]][0]), int(matrix2[i][element[0]][1]))
                point_b = (int(matrix2[i][element[1]][0]), int(matrix2[i][element[1]][1]))
                cv2.line(bframe, point_a, point_b, color[5], 4)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, bframe, 1 - alpha, 0, bframe)
        cv2.imshow('output', bframe)
        time.sleep(0.17)
        k = cv2.waitKey(1)
        if k == 27:
            break;


def sincro(path):
    """

    :param path: path returned by dtw() to temporally align two videos
    :return: list of couple of frame's index
    """
    npath = np.asarray(path)
    index = -1
    old = 0
    a = []
    b = []
    c = []
    for i in range(npath.shape[1]):
        if old == npath[0][i]:
            index = index + 1
        else:
            a.append(index)
            old = npath[0][i]
            index = index + 1
    a.append(index)
    index = -1
    old = 0

    for i in range(npath.shape[1]):
        if old == npath[1][i]:
            index = index + 1
        else:
            b.append(index)
            old = npath[1][i]
            index = index + 1
    b.append(index)
    i = 0
    old = -1

    while (i < len(a) and i < len(b)):
        if a[i] > b[i]:
            if b[i] > old:
                c.append([path[0][a[i]], path[1][a[i]]])
                old = a[i]
            i = i + 1
        elif a[i] < b[i]:
            if a[i] > old:
                c.append([path[0][b[i]], path[1][b[i]]])
                old = b[i]
            i = i + 1
        else:
            c.append([path[0][b[i]], path[1][b[i]]])
            old = b[i]
            i = i + 1
    if len(a) > len(b):
        for h in range(i, len(a)):
            if a[h]> old :
                c.append([path[0][a[h]], path[1][a[h]]])
                old = a[h]
    elif len(a) < len(b):
        for h in range(i, len(b)):
            if b[h]> old:
                c.append([path[0][b[h]], path[1][b[h]]])
                old = b[h]
    c.append(c[len(c)-1])
    c.append(c[len(c)-1])
    return c


def let_me_see_sicro(matrix1, matrix2, path):
    """
    Goal: show two sincro's movements
    :param matrix1: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    :param matrix2: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    :param path: list of couple of frame's index
    """
    for i in range(len(path)):
        bframe = np.zeros((650, 650, 3), np.uint8)
        overlay = bframe
        k = path[i][0]
        l = path[i][1]
        for j in range(matrix1.shape[1]):
            x = int(matrix1[k][j][0])
            y = int(matrix1[k][j][1])
            cv2.circle(overlay, (x, y), 5, (255, 0, 0), -1)
        for j in range(matrix2.shape[1]):
            x = int(matrix2[l][j][0])
            y = int(matrix2[l][j][1])
            cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, bframe, 1 - alpha, 0, bframe)
        time.sleep(0.5)
        cv2.imshow('output', bframe)
        time.sleep(0.5)
        k = cv2.waitKey(1)
        if k == 27:
            break;


def visualize(cost, path, x, y):
    """
    :param cost: cost returned by dtw()
    :param path: path returned by dtw()
    :param x: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    :param y: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    """
    plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')

    plt.plot(path[0], path[1], '-o')  # relation
    #plt.xticks(range(len(x)), x)
    #plt.yticks(range(len(y)), y)
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.axis('tight')
    plt.show()


def get_model(in_str):
    weight =[]
    interest_path = 'move/models/'+in_str+'/interest_point.txt'
    model =csv_to_matrix('move/models/'+in_str+'/model.csv',interest_path)
    with open('move/models/'+in_str+'/weight.csv', 'r') as fin:
        reader = csv.reader(fin)
        for row in reader:
            weight.append(row)

    return model, np.asarray(weight)
