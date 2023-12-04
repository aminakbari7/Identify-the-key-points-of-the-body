import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def main():
    g=0
    flag=False
    cap=cv2.VideoCapture("c1.mp4")
    success,img=cap.read()
    orgimg=img
    dd=[]
    croped=[]
    lmlist=[]
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
             t=img[y1:y2+10,x1:x2+10]
             if t.shape[0]>20 and t.shape[1]>20:
                    ff=str(g)+".jpg"
                    g=g+1
                    cv2.imwrite(filename=ff, img=t)
                 
        ####
if __name__=="__main__":
    main()