import cv2 
# drawing the rectangle
image = cv2.imread('input.png')

if image is not None:

    pt1 = (50, 100)
    pt2 = (500, 500)
    color = (255, 0, 255) # Green color in BGR
    thickness = 5
    cv2.rectangle(image, pt1, pt2, color, thickness)
    cv2.imshow('Image with Rectangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()