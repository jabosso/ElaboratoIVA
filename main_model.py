import argparse
from video import video_to_matrix
from utility_tools import *
from csv_tools import *
import body_dictionary as body_dic
from time_tools import *
import os
from linear_transformation import *
from dtw import *

body = body_dic.body()

app = argparse.ArgumentParser()
print("Insert model video")
app.add_argument("-v", "--video", required=True, help="insert video")
app.add_argument("-e", "--exercise", required=True, help="insert type of video exercise")
args = vars(app.parse_args())
print(args['video'])
matrix = video_to_matrix(args["video"])
ex = args["exercise"]
path = 'move/models/' + ex
try:
    os.makedirs(path +'/complete' )
    os.mkdir(path + '/cycle')
except OSError:
    print('creation of directory failed')


interest_point_path = path + '/interest_point.txt'
matrix = linear_transformation(matrix)
data = create_dataframe(matrix)
data=data['frame'].astype(int)
data=data['x'].astype(int)
data=data['y'].astype(int)
data = add_body_parts(data, body.dictionary)
let_me_see(data)
data = remove_not_interest_point(interest_point_path, data)
matrix_to_csv(data, path+'/complete/', ex)
mid_points = cycle_identify(data)
data = generate_cycle_model(data, mid_points)
data_model = data[0]

matrix_to_csv(data_model, 'move/models/' + ex + '/cycle/', 'model')
 #--------
for i in range(len(data)-1):
    dist, cost, acc, path = dtw(data_model, data[i+1])


