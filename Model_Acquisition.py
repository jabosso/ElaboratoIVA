import argparse
from video import video_to_matrix
from utility_tools import *
from csv_tools import *
from time_tools import *
import os
from linear_transformation import linear_transformation
from dtw import dtw

flag_sampling = True
body = body_dic.body()
app = argparse.ArgumentParser()
app.add_argument("-v", "--video", required=True, help="insert video")
app.add_argument("-e", "--exercise", required=True, help="insert type of video exercise")
args = vars(app.parse_args())
exercise = args['exercise']
input_video = args['video']

path = 'move/models/' + exercise
try:
    os.mkdir(path)
except OSError:
    print('creation of directory failed')
try:
    os.makedirs(path + '/complete')
    os.mkdir(path + '/cycle')
except OSError:
    print('creation of directory failed')
# #--------------------------------------------------------------------------------------------------
matrix, total, fps = video_to_matrix(args["video"])
interest_point_path = path + '/interest_point.txt'
matrix = linear_transformation(matrix)
data = create_dataframe(matrix)
data = normalize_dataFrame(data)
data = add_body_parts(data, body.dictionary)
data = remove_not_interest_point(interest_point_path, data)
matrix_to_csv(data, path + '/complete/', exercise)

path = 'move/models/' + exercise



interest_index = get_interest_point(interest_point_path)
#---------------------------------------------------------------------------------------------------

mid_points = cycle_identify(data)
total = total[int(mid_points[0]):int(mid_points[1])]

data = generate_cycle_model(data, mid_points)
n_body= len(pd.unique(data[0].body_part))
for i in range(len(data)):
    data[i]=shifted(data[i],n_body)
    data[i]=normalize_dataFrame(data[i])
# -----sampling-------
    if flag_sampling :
        data[i] = sampling(data[i])
        data[i] = shifted(data[i], n_body)

    # -----------
data_model = data[0]

#-----------------------------------------------------------------------------------------------------

shape1 = len(pd.unique(data_model['frame']))#number frames of data_model
shape2 = len(data) #number cycle
list_p = np.full((shape1,shape2-1),np.inf) #conterrà sincronizzazioni su ogni ciclo
for i in range(shape2-1):
    path = dtw(data_model, data[i+1])
    p = sincro(path)
    for element in p:
        j = element[0]
        value_ = element[1]
        list_p[j][i] = value_

#------------------------------------------------------------------------------------------------------

total_v = []
total_f = []
med_cs=[]

for i in range(shape1): #for any frame
    distances =[] #vector of cosine distances
    my_list = load_matched_frame(data, data_model, i, list_p, shape2-1)
    dim_v = my_list[0].shape[0] #numero di interest point

    for j in range(len(my_list)):
        try:
            v =vector_load(my_list[j],dim_v)
            distances.append(distance_cosine(v,dim_v,interest_index))
        except:
            _=0
    n_distances = len(distances[0])  # number of distances calculated on single frame
    med_cs.append(calcutate_med_distance(distances, n_distances))
    variances_vec = calculate_variance(distances, n_distances,med_cs,i)
    total_v.append(variances_vec)
    total_f.append(np.full(len(variances_vec),i))
#

med_c = np.asarray(med_cs)
t_f =np.asarray(total_f)
t_f = t_f.reshape((-1))
d_med = pd.DataFrame(t_f,columns=['frame'])
med_c = med_c.reshape((-1))
d_med['medium_dist']=med_c
matrix_to_csv(d_med,'move/models/'+exercise+'/cycle/', 'medium',['frame','medium_dist'])
#------------------------------------------------------------------------------

dat_f = pd.DataFrame(t_f,columns=['frame'])
t_v = np.asarray(total_v)
t_v = t_v.reshape((-1))
dat_f['variance']=t_v
matrix_to_csv(dat_f, 'move/models/'+exercise+'/cycle/','variance',['frame','variance'])

m = np.asarray(data_model)
m= pd.DataFrame(m,columns=['frame','x','y','score','body_part'])
matrix_to_csv(m,'move/models/'+exercise+'/cycle/','model',['frame','x','y','body_part'])