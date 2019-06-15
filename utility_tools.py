import  csv
import numpy as np
import body_dictionary as body_dic

body = body_dic.body()

def csv_to_matrix(path,int_path):
    #input:
        #path:path of file csv with landmarks
        #int_path: path of file txt with interest point
    #output:
            #b: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    a=[]
    interest=[]
    temp =open(int_path,'r')
    for elem in temp:
        interest.append(elem)

    with open(path,'r') as fin :
        reader = csv.reader(fin)
        for row in reader:
            a.append(row)

    a = np.asarray(a)
    a = np.reshape(a,(-1,25,3))
    b = np.zeros(shape=(a.shape[0],a.shape[1],2))
    j=0
    for element in a:
        for body_part in interest:
            part = body_part.replace('\n','')
            i = int(body.dictionary[part])
            if element[i][1] !='' and element[i][2]!='' :
                b[j][i][0] = int(element[i][1])
                b[j][i][1] = int(element[i][2])
            else :
                b[j][i][0] = None
                b[j][i][1] = None
        j = j + 1
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if b[i][j][0]==0 and b[i][j][1]==0 :
                b[i][j][0]=None
                b[i][j][1] = None
    return b

def body_space(body_matrix):
    #input:
        #body_matrix: matrix of body landmarks with dim=[Nframes, Nlandmarks,(x,y)]
    #output:
            #max, min: two tuples for max and min values on x and y
    max_a = np.nanmax(body_matrix, axis=1)
    max = np.nanmax(max_a, axis=0)
    min_a = np.nanmin(body_matrix,axis=1)
    min =np.nanmin(min_a,axis=0)
    return max, min

giovi_path='move/arms_warmup/giovi.csv'
bobo_path ='move/arms_warmup/bobo.csv'
interest_path = 'move/arms_warmup/interest_point.txt'

giovi_matrix = csv_to_matrix(giovi_path,interest_path)
bobo_matrix = csv_to_matrix(bobo_path,interest_path)
maxG, minG = body_space(giovi_matrix)
maxB, minB= body_space(bobo_matrix)
#print(max, min)

