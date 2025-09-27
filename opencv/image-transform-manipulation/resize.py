import cv2

image = cv2.imread('input.png')
#width , height

if image is None:
    print("Could not read the image.")
else: 
    print("Image read successfully.")
    resized_image = cv2.resize(image, (1440, 1080))
    cv2.imshow('Resized Image', resized_image)
    cv2.imwrite('resized_image.png', resized_image)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

