import cv2
import numpy as np

img = cv2.imread('input.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 250, cv2.THRESH_BINARY)


# FIND COUNTOURS
contours , hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# DRAWING CONTOURS
#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)

    corners = len(approx)
    shape_name = "Unidentified"
    if corners == 3:
        shape_name = "Triangle"
    elif corners == 4:
        shape_name = "Quadrilateral"
    elif corners == 5:
        shape_name = "Pentagon"
    elif corners >= 6:
        shape_name = "circle"
    else:
        print("Not detected")

    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 10
    cv2.putText(img, shape_name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))

cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
