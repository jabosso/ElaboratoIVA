from utility_tools import *
from csv_tools import *
import numpy as np
import body_dictionary as body_dic
import time
import cv2 as cv2
from time_tools import *
body = body_dic.body()
color = [(255, 0, 0), (251, 49, 229), (106, 49, 229), (255, 255, 0), (64, 255, 0),
         (0, 128, 255), (255, 128, 0), (128, 0, 255), (255, 0, 255), (255, 0, 128),
         (255, 0, 64), (0, 128, 255), (0, 230, 0), (128, 0, 255), (251, 49, 229),
         (255, 0, 0), (251, 49, 229), (106, 49, 229), (255, 255, 0), (64, 255, 0),
         (0, 128, 255), (255, 128, 0), (128, 0, 255), (255, 0, 255), (255, 0, 128)
         ]
data = csv_to_matrix('ciao.csv')
data=add_body_parts(data,body.dictionary)
data=remove_not_interest_point('move/models/ArmsWarmUp/interest_point.txt',data)

#cycle_identify(data)
a=generate_cycle_model(data,[0,1,2])
print(a[1])
def let_me_see(df):
    """
    Goal: show the movement of the person

    :param matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    """
    for i in range(max(df['frame'])+1):
        bframe = np.zeros((650, 650, 3), np.uint8)
        d=df.loc[df['frame'] == i]
        bframe = body_plot(bframe,d)
        cv2.imshow('output', bframe)
        time.sleep(0.2)
        k = cv2.waitKey(1)
        if k == 27:
            break;

def body_plot(frame,d):
    x = d['x']
    y = d['y']
    for j in range(x.shape[0]):
        cv2.circle(frame, (x.iloc[j], y.iloc[j]), 5, [255, 0, 0], -1)
    for el in body.connection:
        d_A = d.loc[d['body_part'] == body.dictionary[el[0]]]
        d_B = d.loc[d['body_part'] == body.dictionary[el[1]]]
        try:
            _ = d_A['body_part'].item()
            _ = d_B['body_part'].item()
            point_a = (d_A.x.item(), d_A.y.item())
            point_b = (d_B.x.item(), d_B.y.item())
            cv2.line(frame, point_a, point_b, color[el[0]], 4)
        except:
            _ = ''
    return frame

def let_me_see_two_movements(df1, df2):
    for i in range(max(df1['frame'])+1):
        bframe1 = np.zeros((650, 650, 3), np.uint8)
        bframe2 = bframe1
        d1=df1.loc[df1['frame'] == i]
        d2 = df2.loc[df2['frame'] == i]
        bframe1 = body_plot(bframe1,d1)
        bframe2 = body_plot(bframe2, d2)
        alpha = 0.3
        cv2.addWeighted(bframe2, alpha, bframe1, 1 - alpha, 0, bframe1)
        cv2.imshow('output', bframe1)
        time.sleep(0.2)
        k = cv2.waitKey(1)
        if k == 27:
            break;


#let_me_see(data)


#sincro
#let me see_sincro
