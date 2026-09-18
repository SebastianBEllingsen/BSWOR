import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

#image = cv2.imread("images.png")
cap = cv2.VideoCapture("test.mp4")

#capture camera
cam = cv2.VideoCapture(1)

while(True):

    #reads the cam feed
    ret, frame = cap.read()

    results = model(frame, stream=True)
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        #print(xyxyn)
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box
        #shows pngs in browser
        #results.show()

        font = cv2.FONT_HERSHEY_SIMPLEX

        for n in xywh:
            centerx = int(n[0])
            centery = int(n[1])
            width = int(n[2])
            height = int(n[3])

        for m in xyxy:
            xA = int(m[0])
            yA = int(m[1])
            xB = int(m[2])
            yB = int(m[3])

        #for k in confs:
        print("THIS IS", confs)

        

        cv2.rectangle(frame,(xA, yA),(xB, yB),(0,0,0),3)

        for name in names:
            namae = name
            #midxB = xB/2
    
        cv2.putText(frame, namae, (centerx, yA), font, 1, (0,255,255),2, cv2.LINE_4)

        #need to isolate only the conf value
        cv2.putText(frame, str(confs), (centerx, yB), font, 1, (0,255,255),2, cv2.LINE_4)

    if ret:
        #cv2.line(frame, (0, 0), (500, 600), (0, 0, 0), 1)
        cv2.imshow('main',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    else:
        print("could not capture frame")

    

    #print (frame)
    
    #do edits like this
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #then pass in like this
    #cv2.imshow('frame',gray)
