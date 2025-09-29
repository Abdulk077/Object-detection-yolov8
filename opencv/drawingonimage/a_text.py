import cv2 

image = cv2.imread('input.png')

if image is not None:

    cv2.putText(image, 'Abdul', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow('Image with Text', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
