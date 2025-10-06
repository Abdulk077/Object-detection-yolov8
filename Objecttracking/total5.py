import cv2
from ultralytics import YOLO
from collections import Counter
import numpy as np
import pandas as pd

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("cars.mp4")

selected_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
class_names = model.model.names

# Track unique IDs per class
seen_ids_by_class = {class_names[c]: set() for c in selected_classes}
tracker_id_map = {}
next_compact_id = 1
total_objects = 0

# DataFrame to store unique detections only once
columns = ["first_frame", "class_name", "tracker_id", "confidence", "x1", "y1", "x2", "y2"]
detections_df = pd.DataFrame(columns=columns)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
thickness = 2
line_type = cv2.LINE_AA

frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

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

    mask = np.isin(cls, selected_classes)
    boxes, cls, conf, ids = boxes[mask], cls[mask], conf[mask], ids[mask]

    for box, c, cf, track_id in zip(boxes, cls, conf, ids):
        x1, y1, x2, y2 = box
        class_name = class_names[int(c)]

        # Only add to DataFrame if this tracker ID is new for this class
        if track_id not in seen_ids_by_class[class_name]:
            seen_ids_by_class[class_name].add(track_id)
            total_objects += 1

            if track_id not in tracker_id_map:
                tracker_id_map[track_id] = next_compact_id
                next_compact_id += 1
            compact_id = tracker_id_map[track_id]

            # Append unique detection to DataFrame
            detections_df = pd.concat([detections_df, pd.DataFrame([{
                "first_frame": frame_number,
                "class_name": class_name,
                "tracker_id": compact_id,
                "confidence": cf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }])], ignore_index=True)

        # Draw rectangles and labels for current frame
        if track_id not in tracker_id_map:
            tracker_id_map[track_id] = next_compact_id
            next_compact_id += 1
        compact_id = tracker_id_map[track_id]

        label = f"{class_name} #{compact_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), font, 0.6, (0, 255, 0), 2, line_type)

    # Automatic text spacing
    y_offset = 10
    (text_width, text_height), _ = cv2.getTextSize("Test", font, font_scale, thickness)
    line_height = text_height + 10

    # Per-class unique counts
    for cls_name, id_set in seen_ids_by_class.items():
        cv2.putText(frame, f"{cls_name}: {len(id_set)}", (10, y_offset), font, font_scale, (255, 255, 255), thickness, line_type)
        y_offset += line_height

    counts = Counter(cls)
    summary = ", ".join([f"{counts[k]} {class_names[k]}" for k in counts])
    cv2.putText(frame, f"Detected now: {summary}", (10, y_offset), font, font_scale, (255, 255, 255), thickness, line_type)
    y_offset += line_height

    cv2.putText(frame, f"Total unique objects: {total_objects}", (10, y_offset), font, font_scale, (0, 255, 255), thickness, line_type)

    cv2.imshow("YOLOv8 + ByteTrack", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save unique detections overview
detections_df.to_csv("unique_detections_overview.csv", index=False)
print("Saved unique detections overview to unique_detections_overview.csv")
