from utility_tools import *
from csv_tools import *
import numpy as np
import body_dictionary as body_dic
from scipy.spatial.distance import cdist
import time
import cv2 as cv2
from time_tools import *
from dtw import my_dtw
body = body_dic.body()
color = [(255, 0, 0), (251, 49, 229), (106, 49, 229), (255, 255, 0), (64, 255, 0),
         (0, 128, 255), (255, 128, 0), (128, 0, 255), (255, 0, 255), (255, 0, 128),
         (255, 0, 64), (0, 128, 255), (0, 230, 0), (128, 0, 255), (251, 49, 229),
         (255, 0, 0), (251, 49, 229), (106, 49, 229), (255, 255, 0), (64, 255, 0),
         (0, 128, 255), (255, 128, 0), (128, 0, 255), (255, 0, 255), (255, 0, 128)
         ]
datafr='move/models/prova/complete/prova.csv'

data=csv_to_matrix(datafr)
mid_points = cycle_identify(data)
data = generate_cycle_model(data, mid_points)
data_model = data[0]
for i in range(len(data)-1):
    path = my_dtw(data_model, data[i+1])
    df1,df2=sincro_cycle(data_model,data[i+1],path)
    let_me_see_two_movements(df1,df2)