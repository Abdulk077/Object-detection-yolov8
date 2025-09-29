import cv2

image = cv2.imread('input.png')

if image is not None:
    # fliping the image 
    h , w = image.shape[:2] 
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, 360, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    cv2.imshow('Rotated Image', rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
