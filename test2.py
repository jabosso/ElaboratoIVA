from utility_tools import *
from csv_tools import *
import numpy as np
import body_dictionary as body_dic
import time
import cv2 as cv2
from time_tools import *
body = body_dic.body()

data = csv_to_matrix('ciao.csv')
data=add_body_parts(data,body.dictionary)
data=remove_not_interest_point('move/models/ArmsWarmUp/interest_point.txt',data)


