import cv2
from ultralytics import YOLO
from collections import Counter
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("cars.mp4")

# Classes to track (example: vehicles)
selected_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
class_names = model.model.names

# Dictionary to store unique IDs per class
seen_ids_by_class = {class_names[c]: set() for c in selected_classes}

# Dictionary to remap tracker IDs to compact sequential IDs
tracker_id_map = {}
next_compact_id = 1

# Total unique objects counter
total_objects = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Track objects using ByteTrack
    results = model.track(frame, persist=True, conf=0.4, tracker="bytetrack.yaml")
    
    if results[0].boxes is None:
        cv2.imshow("YOLOv8 + ByteTrack", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        continue

    # Convert to NumPy arrays
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    cls = results[0].boxes.cls.cpu().numpy().astype(int)
    conf = results[0].boxes.conf.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy().astype(int)

    # Filter only selected classes
    mask = np.isin(cls, selected_classes)
    boxes, cls, conf, ids = boxes[mask], cls[mask], conf[mask], ids[mask]

    # Draw boxes, labels, and update seen IDs
    for box, c, cf, track_id in zip(boxes, cls, conf, ids):
        x1, y1, x2, y2 = box
        class_name = class_names[int(c)]

        # Update unique IDs per class and total objects
        if track_id not in seen_ids_by_class[class_name]:
            seen_ids_by_class[class_name].add(track_id)
            total_objects += 1  # Increment total objects only when a new ID is seen

        # Map tracker ID to compact sequential ID for visualization
        if track_id not in tracker_id_map:
            tracker_id_map[track_id] = next_compact_id
            next_compact_id += 1
        compact_id = tracker_id_map[track_id]

        label = f"{class_name} #{compact_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display per-class unique counts
    y_offset = 30
    for cls_name, id_set in seen_ids_by_class.items():
        cv2.putText(frame, f"{cls_name}: {len(id_set)}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30

    # Display current frame counts
    counts = Counter(cls)
    summary = ", ".join([f"{counts[k]} {class_names[k]}" for k in counts])
    cv2.putText(frame, f"Detected now: {summary}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30

    # Display total unique objects across all classes
    cv2.putText(frame, f"Total unique objects: {total_objects}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("YOLOv8 + ByteTrack", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
