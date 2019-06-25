import csv
import cv2 as cv2
import time
import math
import numpy as np
import body_dictionary as body_dic
#import matplotlib.pyplot as plt
from csv_tools import *
import pandas as pd
body = body_dic.body()
dict1=['frame','x','y','score']


color = [(255, 0, 0), (251, 49, 229), (106, 49, 229), (255, 255, 0), (64, 255, 0),
         (0, 128, 255), (255, 128, 0), (128, 0, 255), (255, 0, 255), (255, 0, 128),
         (255, 0, 64), (0, 128, 255), (0, 230, 0), (128, 0, 255), (251, 49, 229),
         (255, 0, 0), (251, 49, 229), (106, 49, 229), (255, 255, 0), (64, 255, 0),
         (0, 128, 255), (255, 128, 0), (128, 0, 255), (255, 0, 255), (255, 0, 128)
         ]
'''



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



def let_me_see(df):
    """
    Goal: show the movement of the person

    :param matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    """
    #print max
    for i in range(max(df['frame'])):
        bframe = np.zeros((650, 650, 3), np.uint8)
        overlay = bframe
        if()
        cv2.circle(overlay, (x, y), 5, color[j], -1)

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
'''

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

'''
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

'''
def get_model(exercise):
    # weight =[]
    # interest_path = 'move/models/'+exercise+'/interest_point.txt'
    model =csv_to_matrix('move/models/'+exercise+'/cycle/model.csv')
    weight=csv_to_matrix('move/models/'+exercise+'/cycle/weight.csv')
    # with open('move/models/'+exercise+'/weight.csv', 'r') as fin:
    #     reader = csv.reader(fin)
    #     for row in reader:
    #         weight.append(row)
    return model,weight

def remove_not_interest_point(int_path,data):
    """
    Goal: delete not interest point in data

    :param int_path: path of file txt with interest point of body
    :param data: dataframe [frame;x;y;score;body_part]
    :return: dataframe without not important interest point for the actual exercise
    """
    interest = []
    temp = open(int_path, 'r')
    for elem in temp:
        interest.append(elem.replace('\n', ''))
    for i in range(len(data)):
        bp=data.loc[i,'body_part']
        if not bp in interest:
            data.drop(i,inplace=True)
    return data

def create_dataframe(matrix, dict=dict1):
    """
    Goal: create dataframe without index

    :param matrix: matrix [N*M]
    :param dict: name of columns [M] -default:['frame','x','y','score']
    :return: dataframe without index
    """
    df = pd.DataFrame(data=np.array(matrix), columns=dict)
    blankIndex = [''] * len(df)
    df.index = blankIndex
    return df

def add_body_parts(df,body_part):
    """
    Goal: add new column to dataframe containing body's parts

    :param df: dataframe[frame;x;y;score]
    :param body_part: dictionary containing body's parts
    :return:dataframe with new column of body's parts
    """
    bp =[]
    l = math.ceil(len(df) / 15)  # ritorna l'intero superiore
    for i in range(l):
        for i in range(15):
            bp.append(body_part[i])
    s = pd.Series(bp)
    df['body_part']=(s)
    return df