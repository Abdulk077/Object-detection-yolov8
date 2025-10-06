import cv2
from ultralytics import YOLO
from collections import Counter
import numpy as np
import pandas as pd

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("cars.mp4")

# Classes you want to track (car, motorcycle, bus, truck)
selected_classes = [2, 3, 5, 7]  
class_names = model.model.names

# For unique object tracking
tracker_global_map = {}  # maps raw tracker_id -> compact global ID
next_compact_id = 1
seen_ids_by_class = {class_names[c]: set() for c in selected_classes}
total_unique_objects = 0

# DataFrame to store unique detections
columns = ["first_frame", "class_name", "tracker_id", "confidence", "x1", "y1", "x2", "y2"]
detections_df = pd.DataFrame(columns=columns)

frame_number = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # YOLOv8 tracking
    results = model.track(frame, persist=True, conf=0.4, tracker="bytetrack.yaml")
    if results[0].boxes is None:
        cv2.imshow("YOLOv8 + ByteTrack", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    cls = results[0].boxes.cls.cpu().numpy().astype(int)
    conf = results[0].boxes.conf.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy().astype(int)

    # Filter only selected classes
    mask = np.isin(cls, selected_classes)
    boxes, cls, conf, ids = boxes[mask], cls[mask], conf[mask], ids[mask]

    for box, c, cf, track_id in zip(boxes, cls, conf, ids):
        x1, y1, x2, y2 = box
        class_name = class_names[int(c)]

        # Assign a global compact ID
        if track_id not in tracker_global_map:
            tracker_global_map[track_id] = next_compact_id
            compact_id = next_compact_id
            next_compact_id += 1

            # Store only first appearance in DataFrame
            detections_df = pd.concat([detections_df, pd.DataFrame([{
                "first_frame": frame_number,
                "class_name": class_name,
                "tracker_id": compact_id,
                "confidence": cf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            }])], ignore_index=True)

            # Update per-class and total unique counts
            seen_ids_by_class[class_name].add(compact_id)
            total_unique_objects += 1
        else:
            compact_id = tracker_global_map[track_id]

        # Draw rectangle and label on current frame
        label = f"{class_name} #{compact_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Display per-class counts
    y_offset = 20
    for cls_name, id_set in seen_ids_by_class.items():
        cv2.putText(frame, f"{cls_name}: {len(id_set)}", (10, y_offset), font, 0.7, (255, 255, 255), 2)
        y_offset += 30

    # Display total unique objects
    cv2.putText(frame, f"Total unique objects: {total_unique_objects}", (10, y_offset), font, 0.8, (0, 255, 255), 2)

    # Show current frame
    cv2.imshow("YOLOv8 + ByteTrack", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save clean unique detections overview
detections_df.to_csv("unique_detections_overview.csv", index=False)
print("Saved unique detections overview to unique_detections_overview.csv")
print(f"Total unique objects detected: {total_unique_objects}")
