import math
import numpy as np
from scipy.spatial.distance import cdist
import body_dictionary as body_dic

body = body_dic.body()


def choosen_point(df):
    """
    Goal: determinate the point to max variance

    :param df: dataframe [frame;x;y;score;body_part]
    :return: label of point to max variance
    """
    max_list = []
    max_list_p = []
    for i in body.dictionary:
        try:
            current_part = body.dictionary[i]
            d = df.loc[df['body_part'] == current_part]
            t_list = []
            for j in range(d.shape[0]):
                t_list.append([d.iloc[j].x, d.iloc[j].y])
            m = cdist(t_list, t_list)
            max_list.append(np.max(m[0]))
            max_list_p.append(current_part)
        except:
            _ = ''

    index_max = np.argmax(max_list)
    return max_list_p[index_max]


def cycle_identify(dataframe):
    """
    Goal: find mid_point to determinate cycles

    :param dataframe: [frame;x;y;score;body_part]
    :return: midpoints to generate cycles
    """
    bp = choosen_point(dataframe)
    data = dataframe.loc[dataframe['body_part'] == bp]
    data_tuple = []
    for i in range(data.shape[0]):
        data_tuple.append([data.iloc[i].x, data.iloc[i].y])
    dist = cdist(data_tuple, data_tuple)
    max = np.max(dist[0])
    min = np.min(dist[0])
    threshold = (max + min) / 3
    temp = []
    temp.append(0)
    for i in range(dist.shape[1] - 1):
        if ((dist[0][i] - threshold) * (dist[0][i + 1] - threshold)) < 0:
            temp.append(i)
    temp.append(dist.shape[1])

    midpoints = []
    for i in range(0, len(temp), 2):
        midpoints.append((temp[i ] + temp[i+1]) / 2)
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
        frames = np.arange(start, end + 1)
        df = dataframe.loc[dataframe['frame'].isin(frames)]
        spl_mat.append(df)
    return spl_mat
