import cv2
from ultralytics import YOLO
from collections import Counter

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("test.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model.track(frame , persist=True,conf=0.3, tracker="bytetrack.yaml")
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    cls = results[0].boxes.cls.cpu().numpy().astype(int)
    conf = results[0].boxes.conf.cpu().numpy()

    # Draw boxes & labels
    for box, c, cf in zip(boxes, cls, conf):
        x1, y1, x2, y2 = box
        label = f"{model.model.names[int(c)]} {cf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Count each detected class
    counts = Counter(cls)
    class_names = model.model.names
    summary = ", ".join([f"{v} {class_names[k]}" for k, v in counts.items()])

    # Draw summary box at top-left
    cv2.rectangle(frame, (5, 5), (630, 40), (0, 0, 0), -1)
    cv2.putText(frame, f"Detected: {summary}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

