import math

from ultralytics import  YOLO
clasNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
    "door", "tv", "cell phone", "mouse", "remote", "keyboard", "cell phone", "charger",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush", "pen", "notebook", "marker",
    "wallet", "glasses", "lamp", "file cabinet", "calendar", "speaker", "drone",
    "camera", "binoculars", "calculator", "helmet", "screwdriver", "pliers", "hammer",
    "ladder", "toolbox", "bucket", "lock", "balloon", "tent", "sleeping bag", "flag"
]

import cv2
import cvzone
cap = cv2.VideoCapture(0)
cap.set(3,1080)
cap.set(4,720)
model = YOLO("../RunningYolo/yolov8n.pt")

while True:
    success, img = cap.read()
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w , h = x2-x1, y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            cvzone.putTextRect(img,f'{clasNames[cls]}  {conf}', (max(0,x1),max(20,y1)))

    cv2.imshow("Image",img)
    cv2.waitKey(1)