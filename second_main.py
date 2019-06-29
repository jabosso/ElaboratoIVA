from utility_tools import *
from csv_tools import *
from time_tools import *
from dtw import my_dtw
import numpy as np
from statistic import *
import pandas as pd
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
frame_unique = data_model['frame']
frame_unique = np.asarray(frame_unique)
fram__ = np.zeros((frame_unique.shape[0],3))
for i in range(frame_unique.shape[0]):
    fram__[i][0]=frame_unique[i]
my_dic = ['frame','medium','variance']
new_df = create_dataframe(fram__,my_dic)
new_df =new_df.astype(int)
matrix_to_csv(new_df,'move/models/prova/cycle/','statistic',my_dic)

shape1 = len(pd.unique(data_model['frame']))
shape2 = len(data)-1
list_p = np.full((shape1,shape2),np.inf)
for i in range(len(data)-1):
    path = my_dtw(data_model, data[i+1])
    #list_.app = sincro(path)
    #df1,df2=sincro_cycle(data_model,data[i+1],path)
    #let_me_see_two_movements(df1,df2)
    p = sincro(path)
    for element in p :
        j = element[0]
        value_ = element[1]
        list_p[j][i] = value_

def carico_matched_frame():
    my_list = []
    shift = data_model.iloc[0].frame
    my_list.append(data_model.loc[data_model['frame'] == i + shift])
    for j in range(shape2):
        try:
            current_df = data[j + 1]
            current_frame = list_p[i][j] + current_df.iloc[0].frame
            my_list.append(current_df.loc[current_df['frame'] == current_frame])
        except:
            _ = 0
    return my_list

def calcolo_vect_medio(my_list,dim_v):
    count = 1
    a = my_list[0]
    vect = np.zeros((dim_v, 2))
    for j in range(dim_v):
        tmp =a.iloc[j]
        vect[j][0] += tmp.x
        vect[j][1] += tmp.y
    for j in range(shape2):
        try:
            a = my_list[j+1]
            b = a.iloc[0].frame
            count=count+1
            for h in range(dim_v):
                tmp =a.iloc[h]
                vect[h][0]= vect[h][0]+tmp.x
                vect[h][1]= vect[h][1]+ tmp.y
        except:
            _ = 0
    for j in range(dim_v) :
        vect[j][0] = vect[j][0]/(count)
        vect[j][1] = vect[j][1] / (count)
    return vect

def distance_cos(v,dim):
    m_dist= []
    for j in range(dim-2):
        try:
            current_connection = body.connection[j+1]
            current_dependency = body.dependency[j+1]
            dependency_connection = body.connection[current_dependency]
            point_A= v[current_connection[0]]
            point_B= v[current_connection[1]]
            main_vector = vec(point_A,point_B)
            point_A = v[dependency_connection[0]]
            point_B = v[dependency_connection[1]]
            dependency_vector = vec(point_A, point_B)
            m_dist.append(cosine(main_vector,dependency_vector))
            #print(m_dist)
        except:
            _ =0

    return m_dist

def carico_vettore(df,dim):
    v = np.zeros((dim,2))
    for j in range(dim):
        tmp = df.iloc[j]
        v[j][0]= tmp.x
        v[j][1] = tmp.y
    return v

def calcolo_variance(distances,dim):
    n = len(distances)
    somma = np.zeros(dim-2)
    #print(somma.shape)

    for i in range(n):
        for j in range(dim-2) :
            somma[j] = somma[j] + (distances[0][j]-distances[i][j])**2
    for i in range(dim-2):
        somma[i] = math.sqrt(somma[i]/n)
    return somma

total_v = []
total_f = []
total_vv = []
for i in range(shape1):
    distances =[]

    my_list = carico_matched_frame()
    dim_v =  my_list[0].shape[0]
    vect = calcolo_vect_medio(my_list,dim_v)
    distances.append(distance_cos(vect,dim_v))
    total_vv.append(vect)
    for j in range(len(my_list)-1):
        v =carico_vettore(my_list[j],dim_v)
        distances.append(distance_cos(v,dim_v))
    variances_vec = calcolo_variance(distances,dim_v)
    total_v.append(variances_vec)
    total_f.append(np.full(len(variances_vec),i))
t_f =np.asarray(total_f)
t_f = t_f.reshape((-1))

dat_f = pd.DataFrame(t_f,columns=['frame'])
t_v = np.asarray(total_v)
t_v = t_v.reshape((-1))
dat_f['variance']=t_v
matrix_to_csv(dat_f,'move/models/prova/cycle/','variance',['frame','variance'])
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
matrix_to_csv(dat_v,'move/models/prova/cycle/','mean',['frame','x','y','body_part'])





#----------------------------------------------------------------------------------
     #statistic(df1,df2)


#def sincro_all()

#-----
