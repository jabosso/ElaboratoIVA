import pandas as pd
import numpy as np
import body_dictionary as body_dic
import math

body = body_dic.body()

def matrix_to_csv(matrix,dict,name):
    """

    :param matrix: matrix [N*M]
    :param dict: name of columns [M]
    :param name: name of csv file
    """
    df=pd.DataFrame(data=np.array(matrix),
                   columns=dict)
    blankIndex = [''] * len(df)
    df.index = blankIndex
    df.to_csv(name+'.csv', index=False, columns=dict)

def csv_to_matrix(name):
    """

    :param name: name of csv file
    :return: dataframe, dictionary(name of columns)
    """
    df=pd.read_csv(name)
    #dict = df.columns.to_list()
    #print(dict)
    return df#, dict

def add_body_parts(df,body_part):
    """
    Goal: add new column to dataframe containing body's parts

    :param df: dataframe[frame;x;y;score]
    :param body_part: dictionary containing body's parts
    :return:dataframe with new column of body's parts
    """
    bp =[]
    l = math.ceil(len(df) / 15)  # ritorna l'intero superiore
    for i in range(l):
        for i in range(15):
            bp.append(body_part[i])
    s = pd.Series(bp)
    df['body_part']=(s)
    return df

def remove_not_interest_point(int_path,data):
    """
    Goal: delete not interest point in data

    :param int_path: path of file txt with interest point of body
    :param data: dataframe [frame;x;y;score;body_part]
    :return: dataframe without not important interest point for the actual exercise
    """
    interest = []
    temp = open(int_path, 'r')
    for elem in temp:
        interest.append(elem.replace('\n', ''))
    for i in range(len(data)):
        bp=data.loc[i,'body_part']
        if not bp in interest:
            data.drop(i,inplace=True)
    return data

data = csv_to_matrix('ciao.csv')
data=add_body_parts(data,body.dictionary)
data=remove_not_interest_point('move/ArmsWarmUp/interest_point.txt',data)
print(data)

# for i in range(cf.shape[0]):
#     if cf.loc[i,'frame']==0:
#        cf.loc[i])
# for i in range(0)
