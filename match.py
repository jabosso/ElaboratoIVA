import argparse
from video import video_to_matrix
from linear_transformation import *
from utility_tools import *
import body_dictionary as body_dic

body = body_dic.body()
interest_path = 'move/arms_warmup/interest_point.txt'
action = 'arms_warmup'
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",required=True,help="Video")
args= vars(ap.parse_args())
print('Choose your exercise')
print('1. Arms WarmUp')
print("{}".format(args["video"]))

model , model_w = get_model(action)
print(model.shape, model_w.shape)
matrix_ =video_to_matrix(args["video"])
matrix = matrix_[:, : , 1:]
matrix = funzion(matrix,interest_path)
user_input = linear_transformation(matrix)
