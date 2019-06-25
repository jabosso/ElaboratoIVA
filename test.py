from linear_transformation import *
from utility_tools import *
from scipy.spatial.distance import cdist
from time_tools import *
from dtw import dtw
from statistic import *

giovi_path = 'move/models/ArmsWarmUp/giovi_angle0.csv'
bobo_path = 'move/models/ArmsWarmUp/bobo.csv'
sarra_path = 'move/models/ArmsWarmUp/sarra.csv'
nicco_path = 'move/models/ArmsWarmUp/nicco.csv'
bianca_path = 'move/models/ArmsWarmUp/bianca.csv'
interest_path = 'move/models/ArmsWarmUp/interest_point.txt'

giovi_matrix = csv_to_matrix(giovi_path, interest_path)
bobo_matrix = csv_to_matrix(bobo_path, interest_path)
sarra_matrix = csv_to_matrix(sarra_path, interest_path)
nicco_matrix = csv_to_matrix(nicco_path, interest_path)
bianca_matrix = csv_to_matrix(bianca_path, interest_path)
print(giovi_matrix.shape)
giovi = linear_transformation(giovi_matrix)
bobo = linear_transformation(bobo_matrix)
sarra = linear_transformation(sarra_matrix)
nicco = linear_transformation(nicco_matrix)
bianca = linear_transformation(bianca_matrix)

# giovi,bobo=compare_two_movements(giovi,bobo)
# sarra,nicco=compare_two_movements(sarra,nicco)
nicco, bianca = compare_two_movements(nicco, bianca)

# let_me_see_two_movements(nicco, bianca)

v = cycle_identify(nicco)
spl_mat = []
for i in range(len(v)-1):
    start = int(v[i])
    end =int( v[i+1])
    matr = nicco[start:end]
    spl_mat.append(matr)
print('qui')
print(spl_mat[0].shape)
matrix_to_csv(spl_mat[0],'ArmsWarmUp')
dist, cost, acc, path = dtw(spl_mat[0], spl_mat[1])
#visualize(cost,path,spl_mat[0], spl_mat[1])
path = sincro(path)
#let_me_see(spl_mat[1])
sdc_list =[]
for i in range(len(spl_mat)-1):
    dist, cost, acc, path = dtw(spl_mat[0], spl_mat[i+1])
    path = sincro(path)
    sdc = variance(path,spl_mat[0],spl_mat[i+1])
    sdc_list.append(sdc)
sdc_np = np.asarray(sdc_list)
max_sdc = np.max(sdc_np,axis =0)
for elemnt in max_sdc :
    print(np.arccos(elemnt))
array_to_csv(max_sdc,'arms_warmup_weight')
let_me_see_sicro(spl_mat[0], spl_mat[1], path)
