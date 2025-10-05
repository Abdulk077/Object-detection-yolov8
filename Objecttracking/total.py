import cv2
from ultralytics import YOLO
from collections import Counter

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("test.mp4")

# Sets to store unique IDs
seen_ids = set()  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.5, tracker="bytetrack.yaml")
    
    if results[0].boxes is None:
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    cls = results[0].boxes.cls.cpu().numpy().astype(int)
    conf = results[0].boxes.conf.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy().astype(int)

    # Draw boxes & labels
    for box, c, cf, obj_id in zip(boxes, cls, conf, ids):
        x1, y1, x2, y2 = box
        class_name = model.model.names[int(c)]
        label = f"{class_name} #{obj_id}"

        # Add object ID to seen set
        seen_ids.add(obj_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show total unique objects so far
    cv2.putText(frame, f"Total objects: {len(seen_ids)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # currently detected objects
    counts = Counter(cls)
    print(counts)
    class_names = model.model.names
    summary = ", ".join([f"{v} {class_names[k]}" for k, v in counts.items()])
    cv2.putText(frame, f"Detected: {summary}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("YOLOv8 + ByteTrack", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
