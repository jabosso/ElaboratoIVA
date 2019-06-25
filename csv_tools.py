import pandas as pd
import body_dictionary as body_dic

body = body_dic.body()


def matrix_to_csv(dataframe, path, name):
    """
    Goal: insert dataframe in file csv

    :param dataframe:
    :param path: path where it will be insert csv file
    :param name: name of csv file
    """

    dataframe.to_csv(path + name + '.csv', index=False, columns=dict)


def csv_to_matrix(name):
    """
    Goal: read file csv and export dataframe

    :param name: name of csv file
    :return: dataframe
    """
    df = pd.read_csv(name)
    # dict = df.columns.to_list()
    # print(dict)
    return df  # , dict
