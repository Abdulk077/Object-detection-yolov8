import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv11n-seg model
model = YOLO("yolo11n-seg.pt")  # segmentation model

# Load video
cap = cv2.VideoCapture("test.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Track objects with ByteTrack
    results = model.track(frame, conf=0.3, tracker="bytetrack.yaml", persist=True)

    r = results[0]
    if r.boxes is None:
        continue

    boxes = r.boxes.xyxy.int().cpu().numpy()       # bounding boxes
    ids = r.boxes.id.int().cpu().numpy()           # tracking IDs
    masks = r.masks.data.cpu().numpy() if r.masks else None  # segmentation masks

    for i, obj_id in enumerate(ids):
        x1, y1, x2, y2 = boxes[i]

        # Draw mask if available
        if masks is not None:
            mask = masks[i] > 0.5  # threshold mask
            # resize mask to frame size
            mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0])).astype(bool)
            color = np.array([0, 255, 0], dtype=np.uint8)  # green overlay
            frame[mask] = frame[mask] * 0.5 + color * 0.5

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw tracking ID
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLOv11n-Seg + ByteTrack", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
