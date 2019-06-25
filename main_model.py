import argparse
from video import video_to_matrix
from utility_tools import *
from csv_tools import *
import body_dictionary as body_dic

body = body_dic.body()

app=argparse.ArgumentParser()
print("Insert model video")
app.add_argument("-v","--video",required=True,help="insert video")
args= vars(app.parse_args())
matrix=video_to_matrix(args["video"])
app.add_argument("-e","--exercise",required=True,help="insert type of video exercise")
args= vars(app.parse_args())
ex=args(["exercise"])
path='move/models/'+ex+'/complete/'
interest_point_path='move/models/'+ex+'interest_point.txt'
data=create_dataframe(matrix)
matrix_to_csv(data,path,ex)
data=add_body_parts(data,body.dictionary)
data = remove_not_interest_point(data,interest_point_path)
