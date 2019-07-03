import csv
import cv2 as cv2
import time
import math
import numpy as np
import body_dictionary as body_dic
#import matplotlib.pyplot as plt
from csv_tools import *
import pandas as pd
from statistic import *

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
    n_dot = len(body_part)
    n_columns= df.shape[1]
    l = math.ceil(len(df) / n_dot)  # ritorna l'intero superiore
    for i in range(l):
        for j in range(n_dot):
            bp.append(body_part[j])
            #print('fatto',j)
    df.insert(n_columns,'body_part',bp)
    #print(df)
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

def get_interest_point(int_path):
    """
    Goal: return key correspondent of the interest points

    :param int_path: path of file txt containing the interst points
    :return: vector with keys correspondent of the interest points
    """
    interest = []
    key_interest=[]
    temp = open(int_path, 'r')
    for elem in temp:
        interest.append(elem.replace('\n', ''))
    print(interest)
    keys = list(body.dictionary.keys())
    val = list(body.dictionary.values())
    for label in interest:
        l=keys[val.index(label)]
        key_interest.append(l)
    return key_interest




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
        cv2.circle(blank_frame, (x.iloc[j], y.iloc[j]), 5, color, -1)
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
    for i in range(max(df1['frame'])+ 1):
        bframe1 = np.zeros((650, 650, 3), np.uint8)
        bframe2 = bframe1
        d1 = df1.loc[df1['frame'] == i]
        d2 = df2.loc[df2['frame'] == i]
        red =[0,0,255]
        tic = 7
        bframe1 = body_plot(bframe1, d1)
        bframe2 = body_plot(bframe2, d2,red,tic=6)
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
    count_n =0
    i=0
    while (f[i][0]<0) or (f[i][1]<0):
        count_n+=1
        i+=1
    f=f[count_n:,...]

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
    """
    :param exercise: name of exercise into video
    :return: model and weight
    """
    model = csv_to_matrix('move/models/' + exercise + '/cycle/model.csv')
    weight = csv_to_matrix('move/models/' + exercise + '/cycle/weight.csv')

    return model, weight

def normalize_dataFrame(data):
    """

    :param data: dataframe
    :return: dataframe with integer components: 'frame','x' and 'y'
    """
    data['frame'] = data['frame'].astype(int)
    data['x'] = data['x'].astype(int)
    data['y'] = data['y'].astype(int)
    return data

def shifted(data,n):
    """
    Goal: shift the frames

    :param data: dataframe of cycle
    :param n: number of body part
    :return: dataframe with index wich start from 0 and fram
    """
    shape = data.shape[0]
    new_index=[]
    for i in range(shape):
        new_index.append(i)
    data2 = data.reindex(new_index)
    for i in range(shape):
        data2.iloc[i]=data.iloc[i]
    total=[]
    for i in range(shape//n):
        total.append(np.full(n, i))
    total = np.asarray(total)
    total = total.reshape((-1))
    data2['frame']=total

    return data2


def load_matched_frame(data, model, i, list, shape):
    """
    Goal: create list containing matched frames with model

    :param data: dataframe with any cycle into column
    :param model: dataframe with cycle 0 -model
    :param i: number frame
    :param list: list containing synchronizations of each cycle with model [number_frame_data_model*number cycle-1]
    :param shape: number of cycle less the model
    :return: list containing matched frame of any cycle(included model) with model
    """
    my_list = []
    my_list.append(model.loc[model['frame'] == i])

    for j in range(shape):
        try:
            current_df = data[j + 1]
            current_frame = list[i][j]
            my_list.append(current_df.loc[current_df['frame'] == current_frame])
        except:
            _ = 0

    return my_list

def distance_cosine(v,dim,i_index):
    """

    :param v: matrix [interest point*2]
    :param dim: number of interest point
    :param i_index: correspondent keys of interest point
    :return: vector with cosine distances
    """
    m_dist = []
    for j in range(dim-1):
        try:
            current_connection = body.connection[i_index[j]]
            current_dependency = body.dependency[i_index[j]]
            dependency_connection = body.connection[current_dependency]
            if (dependency_connection[0] in i_index) and (dependency_connection[1] in i_index):
                a_index =i_index.index(current_connection[0])
                b_index = i_index.index(current_connection[1])
                point_A = v[a_index]
                point_B = v[b_index]
                main_vector = vec(point_A, point_B)
                a_index = i_index.index(dependency_connection[0])
                b_index = i_index.index(dependency_connection[1])
                point_A = v[a_index]
                point_B = v[b_index]
                dependency_vector = vec(point_A, point_B)
                m_dist.append(cosine(main_vector, dependency_vector))
        except:
            _ = 0
    return m_dist

def vector_load(df, dim):
    """

    :param df: frames of a cycle
    :param dim: number of interest point
    :return: vector with field x and y
    """
    v = np.zeros((dim, 2))

    for j in range(dim):
        tmp = df.iloc[j]
        v[j][0] = tmp.x
        v[j][1] = tmp.y
    return v

def calculate_variance(distances,dim,med_distance,istance):
    """

    :param distances: cosine distances
    :param dim: number element in any array of distances
    :param med_distance: mean distances
    :param istance: number of frame
    :return: max variance
    """
    print('n')
    print(distances)
    n = len(distances)
    variance = np.zeros(dim)
    old=np.zeros((dim))
    for i in range(n):
        for j in range(dim) :

            if old[j] <math.fabs((med_distance[istance][j] - distances[i][j])):
                old[j]=math.fabs(med_distance[istance][j] - distances[i][j])
        variance=old
    return variance

def calcutate_med_distance(distances, dim):
    """
    GOAL:calculate mean cosine distance on any cycle for ant frame and any interest point

    :param distances: matrix [number_of_cycle*interest_point]
    :param dim: number element in any array of distances
    :return: mean
    """
    n = len(distances)
    med = np.zeros(dim)
    for j in range(dim):
        for i in range(n):
            med[j] = med[j] + distances[i ][j]
        med[j] = med[j] / (n)
    return med

