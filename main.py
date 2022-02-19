import numpy as np
import cv2

ClassName = []
ClassFile = 'coco.names'
with open (ClassFile,'rt')as p: 
    ClassName=p.read().split('\n')

configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightpath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightpath,configpath)
net.setInputSize(320,230)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
###################################################  Image Detection  ################################################### 
# img=cv2.imread("person.png")
# print(img.shape)
# classIds, confs, bbox = net.detect(img,confThreshold = 0.5)
# print(classIds, bbox)

# for classIds, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
#     cv2.rectangle(img, box, color=(230,100,0), thickness=2)
#     cv2.putText(img, ClassName[classIds-1], (box[0]+10, box[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,100,0), thickness=2)

# cv2.imshow("output",img)
# cv2.moveWindow("output",0,0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
##################################################  Video Detection  #################################################
video=cv2.VideoCapture(0)
video.set(3, 740)
video.set(4, 580)
while True: 
        success, img=video.read()
        img = cv2.flip(img, 1)
        classIds, confs, bbox = net.detect(img,confThreshold = 0.5)
        print(classIds, bbox)

        if len(classIds) !=0:
            for classIds, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(230,100,0), thickness=2)
                cv2.putText(img, ClassName[classIds-1], (box[0]+10, box[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,100,0), thickness=2)
                cv2.putText(img, str(round(confidence*100,2)), (box[0]+200, box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,100,0), thickness=2)
        cv2.imshow("output",img)
        cv2.waitKey(1)
        
        


