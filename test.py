from linear_transformation import *
from utility_tools import *


giovi_path = 'move/arms_warmup/giovi_angle0.csv'
bobo_path = 'move/arms_warmup/bobo.csv'
sarra_path = 'move/arms_warmup/sarra.csv'
interest_path = 'move/arms_warmup/interest_point.txt'

giovi_matrix = csv_to_matrix(giovi_path, interest_path)
bobo_matrix = csv_to_matrix(bobo_path, interest_path)
sarra_matrix = csv_to_matrix(sarra_path, interest_path)

giovi = linear_transformation(giovi_matrix)
bobo = linear_transformation(bobo_matrix)
sarra = linear_transformation(sarra_matrix)

let_me_see2(giovi,sarra)