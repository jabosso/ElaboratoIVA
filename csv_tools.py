import pandas as pd
import body_dictionary as body_dic
from utility_tools import create_dataframe


body = body_dic.body()


def matrix_to_csv(dataframe,path,name):
    """

    :param dataframe:
    :param path: path where it will be insert csv file
    :param name: name of csv file
    """

    dataframe.to_csv(path+name+'.csv', index=False, columns=dict)

def csv_to_matrix(name):
    """

    :param name: name of csv file
    :return: dataframe, dictionary(name of columns)
    """
    df=pd.read_csv(name)
    #dict = df.columns.to_list()
    #print(dict)
    return df#, dict





# data = csv_to_matrix('ciao.csv')
# data=add_body_parts(data,body.dictionary)
# data=remove_not_interest_point('move/ArmsWarmUp/interest_point.txt',data)
# print(data)

# for i in range(cf.shape[0]):
#     if cf.loc[i,'frame']==0:
#        cf.loc[i])
# for i in range(0)
