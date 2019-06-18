from linear_transformation import *
from utility_tools import *


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

let_me_see2(nicco,bianca)