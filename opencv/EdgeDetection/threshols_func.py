import cv2

image = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

ret , thresh_img = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)

if image is not None:
    cv2.imshow('Original', image)
    cv2.imshow('Thresholded', thresh_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

