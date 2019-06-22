import numpy as np
from scipy.spatial.distance import cosine

def variance(path,A,B):
    dist_list =[]
    for element in path :
        i = element[0]
        j = element[1]
        dist_list.append(calc_var(A[i],B[j]))

    variance_matrix = np.asarray(dist_list)
    variance_m = np.max(variance_matrix, axis=0)
    return variance_m


def calc_var(A,B):
    connection = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                  (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]
    dist_cos= np.zeros(len(connection))
    for i in range(len(connection)) :
        x= A[connection[i][0]][0]-A[connection[i][1]][0]
        y= A[connection[i][0]][1]-A[connection[i][1]][1]
        z = 0
        vec_A = [x,y,z]
        x = B[connection[i][0]][0] - B[connection[i][1]][0]
        y = B[connection[i][0]][1] - B[connection[i][1]][1]
        vec_B = [x, y, z]

        dist_cos[i] = cosine(vec_A,vec_B)
    return dist_cos



