import pandas as pd
import numpy as np

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
    dict = df.columns.to_list()
    return df, dict

def add_body_parts(df,body_part):
    df['body_part']= pd.Series(np.random.randn(10), index=df.index)
    print(df)

cf, dict = csv_to_matrix('ciao.csv')
for i in range(cf.shape[0]):
    if cf.loc[i,'frame']==0:
       cf.loc[i])
for i in range(0)
