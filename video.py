import pyopenpose as op
import cv2
import numpy as np

def video_to_matrix(src):
    params = dict()
    params["model_folder"] = "models/"
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    cap = cv2.VideoCapture(src)
    datum = op.Datum()
    j = 0
    data_main = []
    k=0
    while (True):
        ret, frame = cap.read()
        dataout = []
        if ret:
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])
            for i in range(15):   #non più 25
                x = int(datum.poseKeypoints[0][i][0])
                y = int(datum.poseKeypoints[0][i][1])
                score = float(datum.poseKeypoints[0][i][2])
                if x == 0 and y == 0:
                    istanza = [j, None, None, None]
                else:
                    istanza = [j, x, y, score]
                dataout.append(istanza)
        else :
            k = 27
        cv2.imshow("prova", datum.cvOutputData)
        if k != 27 :
            k = cv2.waitKey(1)
        j = j + 1
        if k == 27:
            break;
        data_main.append(dataout)
    return np.asarray(data_main)
