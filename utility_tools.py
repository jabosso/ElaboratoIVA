import csv
import cv2 as cv2
import time
import math
import numpy as np
import body_dictionary as body_dic
import matplotlib.pyplot as plt
from csv_tools import *
import pandas as pd

body = body_dic.body()
dict1 = ['frame', 'x', 'y', 'score']

color = [(255, 0, 0), (251, 49, 229), (106, 49, 229), (255, 255, 0), (64, 255, 0), (0, 128, 255), (255, 128, 0),
         (128, 0, 255), (255, 0, 255), (255, 0, 128), (255, 0, 64), (0, 128, 255), (0, 230, 0), (128, 0, 255),
         (251, 49, 229), (255, 0, 0), (251, 49, 229), (106, 49, 229), (255, 255, 0), (64, 255, 0), (0, 128, 255),
         (255, 128, 0), (128, 0, 255), (255, 0, 255), (255, 0, 128)]

def create_dataframe(matrix, dict=dict1):
    """
    Goal: create dataframe without index

    :param matrix: matrix [N*M]
    :param dict: name of columns [M] -default:['frame','x','y','score']
    :return: dataframe without index
    """
    a = len(dict)
    df = pd.DataFrame(data=matrix.reshape(-1,a), columns=dict)
    #blankIndex = [''] * len(df)
    #df.index = blankIndex
    return df


def add_body_parts(df, body_part):
    """
    Goal: add new column to dataframe containing body's parts

    :param df: dataframe[frame;x;y;score]
    :param body_part: dictionary containing body's parts
    :return:dataframe with new column of body's parts
    """
    bp = []
    l = math.ceil(len(df) / 15)  # ritorna l'intero superiore
    for i in range(l):
        for j in range(15):
            bp.append(body_part[j])
    df.insert(4,'body_part',bp)
    return df


def remove_not_interest_point(int_path, data):
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
        bp = data.loc[i].body_part
        if not bp in interest:
            data.drop(i, inplace=True)
    return data


def let_me_see(df):
    """
    Goal: show the movement of the person

    :param df: dataframe [frame;x;y;score;body_part]
    """
    for i in range(max(df['frame']) + 1):
        bframe = np.zeros((650, 650, 3), np.uint8)
        block_frame = df.loc[df['frame'] == i]
        bframe = body_plot(bframe, block_frame)
        cv2.imshow('output', bframe)
        #time.sleep(0.1)
        k = cv2.waitKey(1)
        if k == 27:
            break;


def body_plot(blank_frame, block_frame,color=[0,255,255],tic =4):
    """
    Goal: plot the body of one frame

    :param blank_frame: blank frame to draw
    :param block_frame: dataframe of point of same frame [frame;x;y;score;body_part]
    :return: frame with drawn point and connection
    """
    x = block_frame['x']
    y = block_frame['y']
    for j in range(x.shape[0]):
        cv2.circle(blank_frame, (x.iloc[j], y.iloc[j]), 5, [255, 0, 0], -1)
    for el in body.connection:
        d_A = block_frame.loc[block_frame['body_part'] == body.dictionary[el[0]]]
        d_B = block_frame.loc[block_frame['body_part'] == body.dictionary[el[1]]]
        try:
            _ = d_A['body_part'].item()
            _ = d_B['body_part'].item()
            point_a = (d_A.x.item(), d_A.y.item())
            point_b = (d_B.x.item(), d_B.y.item())
            cv2.line(blank_frame, point_a, point_b, color, tic)
        except:
            _ = ''
    return blank_frame


def let_me_see_two_movements(df1, df2):
    """
    Goal: show the movement of two person simultaneously

    :param df1:dataframe of person 1 [frame;x;y;score;body_part]
    :param df2:dataframe of person 2[frame;x;y;score;body_part]

    """
    for i in range(max(df1['frame'])    + 1):
        bframe1 = np.zeros((650, 650, 3), np.uint8)
        bframe2 = bframe1
        d1 = df1.loc[df1['frame'] == i]
        d2 = df2.loc[df2['frame'] == i]
        red =[0,0,255]
        tic = 7
        bframe1 = body_plot(bframe1, d1)
        bframe2 = body_plot(bframe2, d2,red,tic)
        alpha = 0.7
        risu = cv2.addWeighted(bframe1, alpha, bframe2, 0.3 , 0)
        cv2.imshow('output', risu)
        #time.sleep(0.2)
        k = cv2.waitKey(1)
        if k == 27:
            break;



def sincro(path):
    """

    :param path: path returned by dtw() to temporally align two videos
    :return: list of couple of frame's index
    """
    f = np.asarray(path)
    f = f.transpose()
    f = f[19:]

    c =[]
    old_A  = -1
    old_B = -1
    for i in range(f.shape[0]):
        if (old_A != f[i][0]) and (old_B != f[i][1]):
            coppia = f[i]
            old_A = f[i][0]
            old_B = f[i][1]
            c.append(coppia)
    return c

def sincro_cycle(df1,df2,path):
    p=sincro(path)
    tmp=np.asarray(p)
    a=tmp[...,0]
    b=tmp[...,1]
    for i in range(len(b)):
        b[i]=b[i]+df2.iloc[0].frame
    for i in range(len(b)):
        a[i]=a[i]+df1.iloc[0].frame
    df_t=df1.loc[df1['frame'].isin(a)]
    df_c = df2.loc[df2['frame'].isin(b)]
    frames1 = df_t['frame']
    a =pd.unique(frames1)
    i=0
    df_t.insert(5,'old_frame',df_t['frame'])
    for element in a :
        df_t = df_t.replace({'frame': int(element)},i)

        i=i+1
    frames1 = df_c['frame']
    a = pd.unique(frames1)
    i = 0
    for element in a :
        df_c = df_c.replace({'frame': int(element)},i)
        i=i+1
    return df_t,df_c


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
    model = csv_to_matrix('move/models/' + exercise + '/cycle/model.csv')
    weight = csv_to_matrix('move/models/' + exercise + '/cycle/weight.csv')
    # with open('move/models/'+exercise+'/weight.csv', 'r') as fin:
    #     reader = csv.reader(fin)
    #     for row in reader:
    #         weight.append(row)
    return model, weight
