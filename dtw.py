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



def dtw(x, y,dist=manhattan_distances, warp=1, w= inf, s=1.0):
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
    index=keys[val.index(p)]


    x=x.loc[x['body_part']== p]
    y = y.loc[y['body_part'] == p]
    x=x[['x','y']]
    y=y[['x','y']]
    x1=x.values
    y1=y.values

    x = dist_cos(x1)
    y = dist_cos(y1)

    plt.plot(x)
    plt.plot(y)
    plt.show()
    x = list(np.asarray(x))
    y = list(np.asarray(y))
    print(len(y))
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)


    return D1[-1, -1] / sum(D1.shape), C, D1, path

def my_dtw(x, y):
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

    plt.plot(x)
    plt.plot(y)
    plt.show()
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
    plt.plot(p_n)
    plt.show()
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
