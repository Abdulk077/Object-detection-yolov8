from ultralytics import YOLO
import cv2
model = YOLO('yolov8n.pt')
result = model('images/1.webp', show=True)
annotation_frame = result[0].plot()
cv2.imshow("Detected Image",annotation_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
