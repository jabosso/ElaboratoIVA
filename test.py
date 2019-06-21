from linear_transformation import *
from utility_tools import *
from scipy.spatial.distance import cdist
from time_tools import *
from dtw import dtw

giovi_path = 'move/arms_warmup/giovi_angle0.csv'
bobo_path = 'move/arms_warmup/bobo.csv'
sarra_path = 'move/arms_warmup/sarra.csv'
nicco_path = 'move/arms_warmup/nicco.csv'
bianca_path = 'move/arms_warmup/bianca.csv'
interest_path = 'move/arms_warmup/interest_point.txt'

giovi_matrix = csv_to_matrix(giovi_path, interest_path)
bobo_matrix = csv_to_matrix(bobo_path, interest_path)
sarra_matrix = csv_to_matrix(sarra_path, interest_path)
nicco_matrix = csv_to_matrix(nicco_path, interest_path)
bianca_matrix = csv_to_matrix(bianca_path, interest_path)

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
    print(nicco.shape, matr.shape)
    spl_mat.append(matr)

dist, cost, acc, path = dtw(spl_mat[0], spl_mat[1])
visualize(cost,path,spl_mat[0], spl_mat[1])
path = sincro(path)
#let_me_see(spl_mat[1])
print(path)
let_me_see_sicro(spl_mat[0], spl_mat[1], path)
