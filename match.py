import argparse
from video import video_to_matrix
from linear_transformation import *
from utility_tools import *
from csv_tools import *
import body_dictionary as body_dic
from utility_tools import *

body = body_dic.body()
# interest_path = 'move/ArmsWarmUp/interest_point.txt'
# action = 'ArmsWarmUp'

type_exercise = ['ArmsWarmUp', 'Second']

ap = argparse.ArgumentParser()
ex = 0
# while(ex not in type_exercise):  #per eseguire un eventuale controllo
print('Choose your exercise')
print(type_exercise)
ap.add_argument("-e", "--exercise", required=True, help='Chose exercise')
args = vars(ap.parse_args())
print("{}".format(args["exercise"]))
ex = args["exercise"]
interest_path = 'move/model/' + ex + '/interest_point.txt'

model, model_w = get_model(ex)
# mostra a video il tipo di esercizio da fare
let_me_see(model)

app = argparse.ArgumentParser()
print("Insert you video")
app.add_argument("-v", "--video", required=True, help="Video")  # user video
args = vars(app.parse_args())
print("{}".format(args["video"]))

# print(model.shape, model_w.shape)
matrix = video_to_matrix(args["video"])
user_matrix = linear_transformation(matrix)
# matrix_to_csv(matrix,'user')
data = create_dataframe(user_matrix, ['frame', 'x', 'y', 'score'])
data = add_body_parts(data, body.dictionary)
data = remove_not_interest_point(data, interest_path)
