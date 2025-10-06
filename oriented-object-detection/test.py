from ultralytics import YOLO
import cv2
import numpy as np

# Load pretrained YOLO11n-OBB model
model = YOLO("yolo11n-obb.pt")

# Run inference on a sample image
results = model("https://ultralytics.com/images/boats.jpg")

# Loop through results
for r in results:
    img = r.orig_img.copy()

    # Each detection is an oriented bounding box (OBB)
    for obb, cls_id, conf in zip(r.obb.xyxyxyxy, r.obb.cls, r.obb.conf):
        # Convert to integer coordinates
        pts = np.array(obb.cpu().numpy(), dtype=np.int32).reshape((-1, 1, 2))

        # Draw oriented bounding box
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Get class name and confidence
        label = f"{r.names[int(cls_id)]} {conf:.2f}"

        # Put label above the box
        cv2.putText(
            img, label,
            (pts[0][0][0], pts[0][0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

    # Show image
    cv2.imshow("YOLO Oriented Bounding Boxes", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
