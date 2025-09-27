import cv2 

image = cv2.imread('input.png')

if image is not None:
    flipped_h = cv2.flip(image, 1)
    flipped_v  = cv2.flip(image, 0)
    flipped = cv2.flip(image, -1) #both
    cv2.imshow('Flipped Image', flipped)
    
    cv2.imshow('Flipped Horizontally', flipped_h)
    cv2.imshow('Flipped Vertically', flipped_v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
