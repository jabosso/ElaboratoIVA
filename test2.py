from utility_tools import *
from csv_tools import *
import numpy as np
import body_dictionary as body_dic
import time
import cv2 as cv2
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

def let_me_see(df):
    """
    Goal: show the movement of the person

    :param matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    """
    #print( np.amax(df['frame']) )

    for i in range(max(df['frame'])+1):
        bframe = np.zeros((650, 650, 3), np.uint8)
        overlay = bframe
        d=df.loc[df['frame'] == i]
        x=d['x']
        y=d['y']
        for j in range(x.shape[0]):
            cv2.circle(overlay, (x.iloc[j], y.iloc[j]), 5, [255,0,0], -1)
        cv2.imshow('output',overlay)
        for element in body.connection:
            #da mettere controllo sull'esistenza del punto nel dataframe e quindi della connection
            #if(d.loc[ d['body_part']==body.dictionary[element[0]]]):
            tmp= d.loc[d['body_part']==body.dictionary[element[0]]]
            point_a=(tmp['x'],tmp['y'])
            tmp=d.loc[d['body_part']==body.dictionary[element[1]]]
            point_b=(tmp['x'],tmp['y'])
            cv2.line(bframe, point_a, point_b, color[element[0]], 4)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, bframe, 1 - alpha, 0, bframe)
        cv2.imshow('output', bframe)
        time.sleep(0.2)
        k = cv2.waitKey(1)
        if k == 27:
            break;

let_me_see(data)

#def let_me_see_two_movements(matrix1, matrix2):
#sincro
#let me see_sincro