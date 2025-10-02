import cv2

image = cv2.imread('input.png')

blurred = cv2.medianBlur(image,  7)

cv2.imshow('Original', image)
cv2.imshow('Blurred', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()