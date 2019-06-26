import math
import numpy as np
from scipy.spatial.distance import cdist


def choosen_point(M):
    """
    Goal: determinate the point to max variance

    :param M: dataframe
    :return: point to max variance
    """
    dist = np.zeros((M.shape[1]))
    for i in range(M.shape[1]):
        dist_b = np.zeros((M.shape[0]))
        for j in range(M.shape[0]):
            if not math.isnan(M[j][i][0]) and not math.isnan(M[j][i][1]):
                temp = np.zeros((M.shape[0]))
                for k in range(M.shape[0]):
                    if not math.isnan(M[k][i][0]) and not math.isnan(M[k][i][1]):
                        temp[k] = math.sqrt((M[j][i][0] - M[k][i][0]) ** 2 + (M[j][i][1] - M[k][i][1]) ** 2)
                    else:
                        temp[k] = 0
                dist_b[j] = np.max(temp)
        dist[i] = np.max(dist_b)
    point = np.argmax(dist)
    return point


def cycle_identify(dataframe):
    """
    Goal: find mid_point to determinate cycles

    :param dataframe: [frame;x;y;score;body_part]
    :return: midpoints to generate cycles
    """
    bp = choosen_point(dataframe)
    data=dataframe.loc[dataframe['body_part']==bp ]
    print(data)
    data_tuple=[]
    for i in range(data.shape[0]):
        data_tuple.append([data.iloc[i].x,data.iloc[i].y])
    dist=cdist(data_tuple,data_tuple)
    max=np.max(dist[0])
    min=np.min(dist[0])
    threshold = (max + min) / 3
    temp = []
    temp.append(0)
    for i in range(dist.shape[1] - 1):
        if ((dist[0][i] - threshold) * (dist[0][i + 1] - threshold)) < 0:
            temp.append(i)
    temp.append(dist.shape[1])
    midpoints = []
    for i in range(0, len(temp)-1, 2):
        midpoints.append((temp[i + 1] + temp[i]) / 2)
    return midpoints


def generate_cycle_model(dataframe, midpoints):
    """
    Goal:generate one dataframe for any cycle

    :param dataframe: [frame;x;y;score;body_part]
    :param midpoints: points to split cycles
    :return array of cycles containing dataframes
    """
    spl_mat = []
    for i in range(len(midpoints) - 1):
        start = int(midpoints[i])
        end = int(midpoints[i + 1])
        frames=np.arange(start,end+1)
        df=dataframe.loc[dataframe['frame'].isin(frames)]
        spl_mat.append(df)
    return spl_mat

