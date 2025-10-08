import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd

# Load YOLO model
model = YOLO("yolov8n.pt")  # change to your model file
cap = cv2.VideoCapture("motorbikes-1.mp4")  # your video path

# Classes you want to track (car, motorcycle, bus, truck)
selected_classes = [0, 2, 3, 5, 7]
class_names = model.model.names

# Tracker global map: raw tracker ID -> compact global ID
tracker_global_map = {}
next_compact_id = 1

# Per-class counts: tracker_id -> class-specific count
seen_ids_by_class = {class_names[c]: {} for c in selected_classes}
total_unique_objects = 0

# DataFrame to store unique detections
columns = ["first_frame", "class_name", "tracker_id", "class_count", "confidence", "x1", "y1", "x2", "y2"]
detections_df = pd.DataFrame(columns=columns)

frame_number = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # YOLOv8 tracking
    results = model.track(frame, persist=True, conf=0.3, tracker="bytetrack.yaml")
    r = results[0]
    if r.boxes is None:
        cv2.imshow("YOLOv8 + ByteTrack", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        continue

    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    ids = r.boxes.id.cpu().numpy().astype(int)

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
            total_unique_objects += 1
        else:
            compact_id = tracker_global_map[track_id]

        # Assign per-class count for this object
        if track_id not in seen_ids_by_class[class_name]:
            class_count = len(seen_ids_by_class[class_name]) + 1
            seen_ids_by_class[class_name][track_id] = class_count
            # Store in DataFrame
            detections_df = pd.concat([detections_df, pd.DataFrame([{
                "first_frame": frame_number,
                "class_name": class_name,
                "tracker_id": compact_id,
                "class_count": class_count,
                "confidence": cf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            }])], ignore_index=True)
        else:
            class_count = seen_ids_by_class[class_name][track_id]

        # Draw rectangle and label
        label = f"{class_name} {class_count} #{compact_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Display per-class counts on top-left
    y_offset = 20
    for cls_name, class_dict in seen_ids_by_class.items():
        cv2.putText(frame, f"{cls_name}: {len(class_dict)}", (10, y_offset), font, 0.7, (255, 255, 255), 2)
        y_offset += 30

    # Display total unique objects
    cv2.putText(frame, f"Total unique objects: {total_unique_objects}", (10, y_offset), font, 0.8, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("YOLOv8 + ByteTrack", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save unique detections overview
detections_df.to_csv("unique_detections_overview.csv", index=False)
print("Saved unique detections overview to unique_detections_overview.csv")
print(f"Total unique objects detected: {total_unique_objects}")
