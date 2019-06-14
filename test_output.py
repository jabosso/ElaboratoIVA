import cv2, csv
import numpy as np
import time
saver= []

with open('giovi.csv', mode='r') as csv_file:
    reader =csv.reader(csv_file, delimiter=',')
    for row in reader:        
        saver.append(row)
mypoints = np.asarray(saver)
nframe =mypoints.shape[0] // 25	
connection =[(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
             (1,8),(8,9),(9,10),(10,11),(8,12),(12,13),(13,14)]
for i in range(nframe):  
    oldpoint=(0,0)  
    listpoint=[]
    bframe = np.zeros((800,600,3), np.uint8)
    for j in range(25):         
        if (mypoints[i*25+j][1])!='' :
            temp = mypoints[i*25+j][1] 
       
            x = int(temp.replace("'",''))        
            temp = mypoints[i*25+j][2]
            y = int(temp.replace("'",'')) 
            point= (x,y)   
            listpoint.append(point)    
            cv2.circle(bframe,point,3,(255,124,0),-1) 
            font = cv2.FONT_HERSHEY_SIMPLEX
        else:
            listpoint.append([None,None])       
            #cv2.putText(bframe,str(j),           (point[0]+10,point[1]+10),fontFace=font,fontScale=0.7,color=(255,124,0))
    for element in connection:         
        if listpoint[element[0]][0] != None and listpoint[element[1]][0] !=None:
            cv2.line(bframe,listpoint[element[0]],listpoint[element[1]],(255,124,0))         
    cv2.imshow('output',bframe)
    time.sleep(0.17)
    k = cv2.waitKey(1)
    if k==27:
        break;


    
        
            

