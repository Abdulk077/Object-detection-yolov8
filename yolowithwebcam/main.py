import math

from ultralytics import  YOLO

import cv2
import cvzone
cap = cv2.VideoCapture(0)
cap.set(3,1080)
cap.set(4,720)
model = YOLO("yolov8l.pt")
class_names = model.model.names
img = cv2.imread("1.webp")
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

            cvzone.putTextRect(img,f'{class_names[cls]}  {conf}', (max(0,x1),max(20,y1)))

    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()