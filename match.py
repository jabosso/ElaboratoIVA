import argparse
from video import video_to_matrix
from linear_transformation import *
from utility_tools import *
from csv_tools import *
import body_dictionary as body_dic
from time_tools import *
from dtw import dtw
#from second_main import carico_vettore, distance_cos
import time


body = body_dic.body()
# interest_path = 'move/ArmsWarmUp/interest_point.txt'
# action = 'ArmsWarmUp'

type_exercise = ['ArmsWarmUp', 'prova']

#ap = argparse.ArgumentParser()
ex = 0
# while(ex not in type_exercise):  #per eseguire un eventuale controllo
#print('Choose your exercise')
#print(type_exercise)
#ap.add_argument("-e", "--exercise", required=True, help='Chose exercise')
#ap.add_argument("-v", "--video", required=True, help="Video")  # user video
#args = vars(ap.parse_args())
#print("{}".format(args["exercise"]))
#ex = args["exercise"]
ex= 'Arms2'
# vid_na = 'data/model_arm.mp4'
interest_path = 'move/models/' + ex + '/interest_point.txt'

#model_path = 'move/models/'+ex+'/cycle/mean.csv'
model_path = 'move/models/'+ex+'/cycle/model.csv'

model = csv_to_matrix(model_path)
model['frame']= model['frame'].astype((int))
model['x']= model['x'].astype(int)
model['y']= model['y'].astype(int)
interest_index = get_interest_point(interest_path)


#----tutto comm
#model, model_w = get_model(ex)
# mostra a video il tipo di esercizio da fare
#let_me_see(mean_model)
#print("{}".format(args["video"]))

# print(model.shape, model_w.shape)
#matrix = video_to_matrix(args["video"])
# matrix_to_csv(matrix,'user')
#-----


#--- comm Gio

# matrix = video_to_matrix(vid_na)
#user_matrix = linear_transformation(matrix)
#data = create_dataframe(user_matrix, ['frame', 'x', 'y', 'score'])

# data = add_body_parts(data, body.dictionary)
# data = remove_not_interest_point(interest_path, data)
# data['frame']=data['frame'].astype(int)
# data['x']=data['x'].astype(int)
# data['y']=data['y'].astype(int)
#-----end
data=csv_to_matrix('move/models/Arms2/complete/Arms2.csv')
mid_points = cycle_identify(data)

data_c = generate_cycle_model(data, mid_points)
n_body= len(pd.unique(data_c[0].body_part))
for i in range(len(data_c)):
    data_c[i]=shifted(data_c[i],n_body)
    data_c[i]=normalize_dataFrame(data_c[i])

shape1 = len(pd.unique(model['frame']))
shape2 = len(data_c)
list_p = np.full((shape1,shape2),np.inf)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,0,255)
#lineType               = 2

for i in range(shape2):
    path = dtw(model, data_c[i])
    p = sincro(path)
    for element in p:
        j = element[0]
        value_ = element[1]
        list_p[j][i] = value_
dim_v = len(pd.unique(model['body_part']))

variance_w =csv_to_matrix('move/models/'+ex+'/cycle/variance.csv')
medium_cs = csv_to_matrix('move/models/'+ex+'/cycle/medium.csv')
color1= [0,0,255]
color2 = [0,255,0]
for i in range(shape2):
    total_score = 0
    vect=[]
    current_cycle = data_c[i]
    for j in range(shape1):
        if list_p[j][i] != np.inf:
            vect.append([j,list_p[j][i]])
    for elem in vect:
        f_u = current_cycle.loc[current_cycle['frame'] == elem[1]]
        v_u = vector_load(f_u,dim_v)
        distance_user =distance_cosine(v_u,dim_v,interest_index)
        f_m=f_u
        v_m = vector_load(f_m,dim_v)
        distance_model =distance_cosine(v_m,dim_v,interest_index)
        current_variance = variance_w.loc[variance_w['frame'] == elem[0]]

        current_medium = medium_cs.loc[medium_cs['frame'] == elem[0]]
        varia = current_variance['variance']

        medium_dist = current_medium['medium_dist']
        result = np.zeros(len(varia))
        check = np.zeros(len(varia))
        worse = np.zeros(5)

        for i in range(len(varia)):
            result[i] = math.fabs(distance_user[i]-medium_dist.iloc[i])
            if (round(result[i],3)<round(varia.iloc[i],3)) or (round(result[i],3)== round(varia.iloc[i],3)) :
                check[i]=1

        for element in check:
            total_score = total_score+element
        bframe = np.zeros((650,650,3),np.uint8)
        bframe = body_plot(bframe,f_m,color1,tic=8)
        bframe = body_plot(bframe, f_u, color2)
        cv2.putText(bframe,str(check),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor)
        cv2.imshow('confronto',bframe)
        k = cv2.waitKey(1)
        time.sleep(0.3)
        if k == 27:
            break;
    total_score= total_score/(len(varia)*len(vect))
    print('la precisione del ciclo  è ',total_score)














