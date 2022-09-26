import mediapipe as mp
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math

class PoseDetection():


    def __init__(self):
        
        self.mdraw = mp.solutions.drawing_utils
        self.mpose = mp.solutions.pose
        self.pose = self.mpose.Pose()
        
    def finddetect(self,img,draw=True,connect=True):
        
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB)
        
        if self.result.pose_landmarks:
            if draw:
                if connect:
                    self.mdraw.draw_landmarks(img,self.result.pose_landmarks,self.mpose.POSE_CONNECTIONS)
                else:
                    self.mdraw.draw_landmarks(img,self.result.pose_landmarks)
        
        return img


    def findposition(self,img,draw=True):
        
        self.hms = []
        
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x * w), int(lm.y * h)        
                self.hms.append([id,cx,cy])
                
                if draw:
                    cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
                
        return self.hms
    
    def findangle(self,img,p1,p2,p3,draw=True):
        
        x1,y1 = self.hms[p1][1:]
        x2,y2 = self.hms[p2][1:]
        x3,y3 = self.hms[p3][1:]
        
        angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y2-y1,x2-x1))
        
        if angle < 0:
            if angle > -180:
                angle *= -1
            else:
                angle += 180
        
        if draw:
            cv2.line(img,(x2,y2),(x1,y1),(255,255,255),4)
            cv2.line(img,(x3,y3),(x2,y2),(255,255,255),4)
            
            cv2.circle(img,(x1,y1),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x1,y1),15,(0,0,255),2)
            cv2.circle(img,(x2,y2),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(0,0,255),2)
            cv2.circle(img,(x3,y3),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x3,y3),15,(0,0,255),2)
            cv2.putText(img,str(int(angle)),(x2-20,y2+50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        
        return int(angle)
        

def main():
    
    dir = "C:/Users/Rachit/Downloads/"
    video = "ai_trx.mp4"
    path = os.path.join(dir,video)
    
    cap = cv2.VideoCapture(path)

    ptime = 0
    ctime = 0
    
    pose = PoseDetection()
    
    while True:
        ret,frame = cap.read()
        frame = pose.finddetect(frame)
        lms = pose.findposition(frame)
        
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        cv2.putText(frame,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow("the frame",frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()