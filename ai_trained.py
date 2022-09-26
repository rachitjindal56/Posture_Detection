import mediapipe as mp
import cv2
import posture_detection as pd
import os
import time
import numpy as np
import math

dir = "C:/Users/Rachit/Downloads/"
video = "ai_trx.mp4"

path = os.path.join(dir,video)

cap = cv2.VideoCapture(path)

ptime = 0
ctime = 0
bar = 0
pe = 0
pose = pd.PoseDetection()

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(1280,720))
    frame = pose.finddetect(frame,False)
    lms = pose.findposition(frame,False)
    
    if len(lms) > 0:
        angle1 = pose.findangle(frame,11,13,15)
        angle2 = pose.findangle(frame,12,14,16)
        
        pe = np.interp(angle2,(5,105),(0,100))
        bar = np.interp(angle2,(5,105),(700,300))

        
    cv2.rectangle(frame,(1080,300),(1125,700),(0,255,0),2)
    cv2.rectangle(frame,(1080,int(bar)),(1125,700),(0,255,0),cv2.FILLED)
    cv2.putText(frame,str(int(pe))+str("%"),(1080,250),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(frame,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.imshow("the frame",frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()