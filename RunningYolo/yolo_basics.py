from ultralytics import YOLO
import cv2
model = YOLO('yolov8n.pt')
class_names = model.model.names
results = model('images/1.webp', show=True)
print("results", results)
boxes = results[0].boxes.xyxy.numpy()
confs = results[0].boxes.conf.numpy()*100
classes = results[0].boxes.cls.numpy()
print("boxes" ,boxes, confs,  classes)

annotation_frame = results[0].plot()
cv2.imshow("Detected Image",annotation_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
