import cv2
import mediapipe as mp
import os
import time
import math
import handdetection_ai as hand

cap = cv2.VideoCapture(0)
cap.set(3,1200)
cap.set(4,720)

detector = hand.handDetection(detectionCon=0.75)

while True:
    ret,img = cap.read()
    img = cv2.flip(img,1)
    
    img = detector.findhands(img)
    lmslist = detector.findposition(img,0,8)
    
    if len(lmslist) > 0:
        
        x1,y1 = lmslist[8][1:]
        x2,y2 = lmslist[12][1:]
    
    cv2.imshow("The frame",img)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()