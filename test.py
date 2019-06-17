from linear_transformation import *
from utility_tools import *


giovi_path = 'move/arms_warmup/giovi_angle0.csv'
bobo_path = 'move/arms_warmup/bobo.csv'
interest_path = 'move/arms_warmup/interest_point.txt'

giovi_matrix = csv_to_matrix(giovi_path, interest_path)
bobo_matrix = csv_to_matrix(bobo_path, interest_path)
poldo = rotation_function(giovi_matrix,(1, 5))
maxG, minG = body_space(poldo)
maxB, minB = body_space(bobo_matrix)
poldo, box_dim = shift_function(poldo, maxG, minG)
poldo = scale_function(poldo,box_dim)
print(poldo[0][1][1], poldo[0][5][1])
let_me_see(poldo)