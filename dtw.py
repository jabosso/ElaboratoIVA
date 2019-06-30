from numpy import array, zeros, full, argmin, inf
from math import isinf
import math
from time_tools import *
import body_dictionary as body_dic
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import manhattan_distances

body = body_dic.body()

def dist_cos(v):
    a=[]
    for element in v :
        a.append(cosine(element,[1,1]))
    return a


def dtw(x, y):
    """
        Computes Dynamic Time Warping (DTW) of two sequences.
        :param array x: N1*M array
        :param array y: N2*M array
        :param int warp: how many shifts are computed.
        :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
        :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
        Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
        """
    p = choosen_point(x)
    keys = list(body.dictionary.keys())
    val = list(body.dictionary.values())
    index = keys[val.index(p)]
    x = x.loc[x['body_part'] == p]
    y = y.loc[y['body_part'] == p]
    x = x[['x', 'y']]
    y = y[['x', 'y']]
    x1 = x.values
    y1 = y.values
    x = dist_cos(x1)
    y = dist_cos(y1)

    #plt.plot(x)    #plt.plot(y)    #plt.show()

    x = np.asarray(x)
    y = np.asarray(y)
    d= np.zeros((x.shape[0],y.shape[0]))
    for i in range(x.shape[0]):
        d[i][0] =math.fabs(x[i]-y[0])
    for i in range(y.shape[0]):
        d[0][i] = math.fabs(x[0] - y[i])
    for i in range(1,x.shape[0]):
            for j in range(1,y.shape[0]):
                d[i][j]= math.fabs(x[i]-y[j])+ min(3*d[i-1,j-1],
                                                   d[i-1,j],
                                                   d[i,j-1])
    path = _traceback(d)
    p_n = np.asarray(path)

    #plt.plot(p_n)    #plt.show()
    return p_n







def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)
