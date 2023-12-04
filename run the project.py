import cv2
import mediapipe as mp
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
croplist=[]
class posedetectors():
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()
    def findpose(self,img,draw=True):
        imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgrgb)  
        p=self.results.pose_landmarks
        if p:
             if draw:
                 self.mp_drawing.draw_landmarks(img, self.results.pose_landmarks,self.mp_pose.POSE_CONNECTIONS)
    def getposition(self,img,draw=True): 
     lmlist=[]   
     if self.results.pose_landmarks:  
       for id,lm in enumerate(self.results.pose_landmarks.landmark):
           h,w,c=img.shape
           cx,cy=int(lm.x*w),int(lm.y*h)
           lmlist.append([cx,cy])
     return lmlist 
def main():
    maxhash=8
    tc=0
    fc=0
    cap=cv2.VideoCapture("c1.mp4")
    video = cap
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'),14, size)
    target1=imagehash.average_hash(Image.open('3.jpg'))
    target2=target1
    target3=target1
    target4=target1
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    flagtarget=False
    flag=False
    detector=posedetectors()
    while True: 
        success,img=cap.read()
        if success==False:
            break;
        orgimg=img
        lmlist=[]
        ####
        image=img
        classes = None
        with open('coco.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        Width = image.shape[1]
        Height = image.shape[0]
        net = cv2.dnn.readNet('2.weights', 'yolov3.cfg')
        net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        outs = net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
          for detection in out:
           scores = detection[5:]
           class_id = np.argmax(scores)
           confidence = scores[class_id]
           if confidence > 0.1:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])            
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
        minh=1000
        for i in indices:
         box = boxes[i]
         if class_ids[i]==0:
             x1=round(box[0])
             y1=round(box[1])
             x2=round(box[0]+box[2])
             y2=round(box[1]+box[3])
             if x1<0:
                x1=0  
             if x2<0:
                x2=0 
             if y1<0:
                y1=0 
             if y2<0:
                y2=0 
             cropped=img[y1:y2,x1:x2]
             if cropped.shape[0]>100 and cropped.shape[1]>100:
                    im_pil = Image.fromarray(cropped)
                    hash1 = imagehash.average_hash(im_pil) 
                    re=abs(target1-hash1)
                    if minh>re and re<maxhash: 
                            #if abs(hash1-target2)<maxhash and abs(hash1-target3)<maxhash:
                            if abs(hash1-target1)<maxhash and abs(hash1-target2)<maxhash+1:
                               print(abs(hash1-target2),abs(hash1-target2 ))
                               minh=re
                               im_np = np.asarray(im_pil)
                               target4=target3
                               target3=target2
                               target2=target1
                               target1=hash1
                               img=cropped
                               flag=True
                               tc+=1
        ####
        fc+=1
        if flag==True:
         detector.findpose(img)
         lmlist=detector.getposition(img) 
         if len(lmlist)!=0:
            if abs(lmlist[15][0]-lmlist[25][0])<50 and abs(lmlist[15][1]-lmlist[25][1])<50:
               cv2.putText(orgimg,"hand to knee",(70,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
            if(lmlist[12][1]-lmlist[14][1])>10 or (lmlist[11][1]-lmlist[13][1])>10:
                cv2.putText(orgimg,"hand up",(70,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        result.write(orgimg)
        #print(fc,tc)
        cv2.imshow("a",orgimg)
        path = 'images'
        ff=str(fc)+".jpg"
        cv2.imwrite(os.path.join(path,ff) ,orgimg)
        flag=False
        if cv2.waitKey(1) == ord('q'):
            break
if __name__=="__main__":
    main()