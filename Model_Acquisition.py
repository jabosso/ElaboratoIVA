import argparse
from video import video_to_matrix
from utility_tools import *
from csv_tools import *
from time_tools import *
import os
from linear_transformation import linear_transformation
from dtw import dtw

body = body_dic.body()
app = argparse.ArgumentParser()
app.add_argument("-v", "--video", required=True, help="insert video")
app.add_argument("-e", "--exercise", required=True, help="insert type of video exercise")
args = vars(app.parse_args())
exercise = args['exercise']
input_video = args['video']
print('You choose '+exercise+ ' as exercise')
print('You choose '+input_video+' as model video to acquire')
#--------------------------------------------------------------------------------------------------

path = 'move/models/' + exercise
try:
    os.mkdir(path )
except OSError:
    print('creation of directory failed')
try:
    os.makedirs(path +'/complete' )
    os.mkdir(path + '/cycle')
except OSError:
    print('creation of directory failed')
#--------------------------------------------------------------------------------------------------

matrix = video_to_matrix(args["video"])
interest_point_path = path + '/interest_point.txt'

matrix = linear_transformation(matrix)
data = create_dataframe(matrix)
data = normalize_dataFrame(data)
data = add_body_parts(data, body.dictionary)
data = remove_not_interest_point(interest_point_path, data)
matrix_to_csv(data, path+'/complete/', exercise)
interest_index = get_interest_point(interest_point_path)
#---------------------------------------------------------------------------------------------------

mid_points = cycle_identify(data)
data = generate_cycle_model(data, mid_points)
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
total_vv = []
med_cs=[]
for i in range(shape1):
    distances =[] #vector of cosine distances
    my_list = load_matched_frame(data, data_model, i, list_p, shape2-1)
    dim_v = my_list[0].shape[0] #numero di interest point
    vect = mean_vector(my_list,dim_v, shape2-1)
    distances.append(distance_cosine(vect,dim_v,interest_index)) #mean distance cosine
    n_distances=len(distances[0]) #number of distances calculated on single frames
    total_vv.append(vect)
    for j in range(len(my_list)):
        try:
            v =vector_load(my_list[j],dim_v)
            distances.append(distance_cosine(v,dim_v,interest_index))
            print('ho cicli sincronizzati per ', i)
        except:
            print('non ho cicli sincronizzati per ',i)

    med_cs.append(calcutate_med_distance(distances, n_distances))
    variances_vec = calculate_variance(distances, n_distances)
    total_v.append(variances_vec)
    total_f.append(np.full(len(variances_vec),i))
med_c = np.asarray(med_cs)
#print(med_c)
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
matrix_to_csv(dat_f,'move/models/'+exercise+'/cycle/','variance',['frame','variance'])
t_vv = np.asarray(total_vv)
t_g =np.zeros((t_vv.shape[0]*t_vv.shape[1]))
for i in range (t_vv.shape[0]):
    for j in range (t_vv.shape[1]):
        t_g[i*t_vv.shape[1]+j]=i

t_vv = t_vv.reshape((t_vv.shape[0]*t_vv.shape[1],2))
dat_v =pd.DataFrame(t_g,columns=['frame'])
dat_v['x']=t_vv[...,0]
dat_v['y']=t_vv[...,1]
dat_v['frame'].astype(int)
dict2 = pd.unique(my_list[0].body_part)
data_v = add_body_parts(dat_v,dict2)
matrix_to_csv(dat_v,'move/models/'+exercise+'/cycle/','mean',['frame','x','y','body_part'])

