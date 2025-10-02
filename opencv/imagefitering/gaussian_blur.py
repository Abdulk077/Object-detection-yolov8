import cv2
# blur effect using Gaussian filter
# take file input from user 



image = cv2.imread('input.png')

blurred = cv2.GaussianBlur(image, (5, 5), 5)

cv2.imshow('Original', image)
cv2.imshow('Blurred', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
