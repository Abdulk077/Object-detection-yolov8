import cv2
# drawing a line
# coose image path
IMAGE_PATH = 'input.png'
image = cv2.imread(IMAGE_PATH)

if image is not None:
    pt1 = (50, 100)
    pt2 = (300, 100)
    color = (255, 0, 255) # Green color in BGR
    thickness = 5
    cv2.line(image, pt1, pt2, color, thickness)
    cv2.imshow('Image with Line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()