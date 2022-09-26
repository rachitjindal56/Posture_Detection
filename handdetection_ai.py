import mediapipe as mp
import cv2
import time

class handDetection():
    
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5):       
        
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
         
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.maxHands,self.detectionCon)
        self.mdraw = mp.solutions.drawing_utils



    def findhands(self,img,draw = True):
        
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
            
        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                if draw:
                    self.mdraw.draw_landmarks(img,handlms,self.mphands.HAND_CONNECTIONS)
                    
        return img
    
    
    def findposition(self,img,handn0=0,finger=-1,draw=True):
            
        lms = []
        
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handn0]
            
            for id,lm in enumerate(myhand.landmark):                        
                h,w,c = img.shape
                cx,cy = int(lm.x * w), int(lm.y * h)
                lms.append([id,cx,cy])        
                
                if draw:
                    if finger == -1:
                        cv2.circle(img,(cx,cy),12,(255,0,255),cv2.FILLED)
                    else:
                        if id == finger:
                            cv2.circle(img,(cx,cy),12,(255,0,255),cv2.FILLED)
                    
        return lms
                

def main():
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    hand = handDetection()
    
    ptime = 0
    ctime = 0 

    while True:
        ret,frame = cap.read()
        frame = hand.findhands(frame)
        lms = hand.findposition(frame)
        
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        
        cv2.putText(frame,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    
        cv2.imshow("The Frame",frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()
    
