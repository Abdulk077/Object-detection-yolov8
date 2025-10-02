import cv2

image = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
# detecting edges of the image using Canny edge detector
edges = cv2.Canny(image, 50, 150)
if edges is not None:
    cv2.imshow('Original', image)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
