import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from body_dictionary import body

bod = body()
from csv_tools import *


def variance(path, A, B):
    dist_list = []
    for element in path:
        i = element[0]
        j = element[1]
        dist_list.append(calc_var(A[i], B[j]))

    variance_matrix = np.asarray(dist_list)
    variance_m = np.max(variance_matrix, axis=0)
    return variance_m


def vec(point_a, point_b):
    vect = (point_a[0] - point_b[0], point_a[1] - point_b[1], 0)
    return vect


def calc_var(A, B):
    dist_cos = np.zeros(len(body.connection) + 1)
    i = -1
    for element in body.connection:
        i = i + 1
        try:
            d0_A = A.loc[A['body_part'] == body.dictionary[element[0]]]
            d1_A = A.loc[A['body_part'] == body.dictionary[element[1]]]
            point0_a = [d0_A['x'].item(), d0_A['y'].item()]
            point0_b = [d1_A['x'].item(), d1_A['y'].item()]
            d0_B = B.loc[B['body_part'] == body.dictionary[element[0]]]
            d1_B = B.loc[B['body_part'] == body.dictionary[element[1]]]
            point1_a = [d0_B['x'].item(), d0_B['y'].item()]
            point1_b = [d1_B['x'].item(), d1_B['y'].item()]
            vec1 = vec(point0_a, point0_b)
            vec2 = vec(point1_a, point1_b)
            dist_cos[i] = cosine(vec1, vec2)
        except:
            dist_cos[i] = None
            _ = 0
    return dist_cos

    return dist_cos


def statistic(model, secondary):
    q = csv_to_matrix('move/models/prova/cycle/statistic.csv')

    for i in pd.unique(model['frame']):
        varianza = calc_var(model.loc[model['frame'] == i], secondary.loc[secondary['frame'] == i])
        dica = pd.Series(varianza)
        dica = dica.dropna()
        r = q.index[q['frame'] == i]
        for j in range(len(r) - 1):
            i = r[j]
            value = dica[j]
            q.at[i, 'variance'] = value
