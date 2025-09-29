import cv2

image = cv2.imread('input.png')

if image is not None:
    h , w = image.shape[:2]
    cv2.circle(image, (w//2, h//2), 200, (255, 0, 255), 1)
    # image , circle centre , radius , color , thickness
    cv2.imshow('Image with Circle', image)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
