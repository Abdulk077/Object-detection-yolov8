from ultralytics import YOLO
import cv2

# Load segmentation model
model = YOLO("yolo11n-seg.pt")  

# Batch inference
results = model([
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg"
])

# Loop over results for each image
for i, r in enumerate(results):
    print(f"Image {i} masks:", r.masks)  # Print mask info

    # Plot bounding boxes + masks
    frame = r.plot()  

    # Show image
    cv2.imshow(f"Segmentation {i}", frame)
    cv2.waitKey(0)  # Wait for a key press to move to the next image

cv2.destroyAllWindows()  # Close all windows after loop
