import cv2
import numpy as np

image = cv2.imread('input.png')

sharpen_kernel = np.array([[0, -1, 0],
                           [-1,  5, -1], 
                              [0, -1, 0]
                              ])

sharpened = cv2.filter2D(image, -1, sharpen_kernel)
cv2.imshow('Original', image)
cv2.imshow('Sharpened', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()


