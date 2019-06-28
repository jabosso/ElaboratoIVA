from utility_tools import *
from csv_tools import *
import numpy as np
import body_dictionary as body_dic
import time
import cv2 as cv2
from time_tools import *
body = body_dic.body()




for i in range(len(body.dependency)):
    print(body.connection[i],body.dependency[i])