import pyopenpose as op
import argparse
import cv2
import numpy as np

color = (255, 0, 0)
parser = argparse.ArgumentParser()
args = parser.parse_known_args()
params = dict()
params["model_folder"] = "models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
cap = cv2.VideoCapture(0)
datum = op.Datum()
while (True):
    ret, frame = cap.read()

    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    for i in range(25):
        print(datum.poseKeypoints[0][i])
        cv2.circle(frame, (int(datum.poseKeypoints[0][i][0]), int(datum.poseKeypoints[0][i][1])), 8, (255, 0, 0), -1)

    cv2.imshow("prova", datum.cvOutputData)
    k = cv2.waitKey(1)
    if k == 27:
        break;

#

#
