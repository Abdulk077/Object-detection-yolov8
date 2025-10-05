import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

#cap = cv2.VideoCapture(0)
frame = cv2.imread('1.webp')

while True:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face Detected', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
    if len(eyes) >= 0:
        cv2.putText(frame, 'Eyes Detected', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
    if len(smiles) >= 1:
        cv2.putText(frame, 'Smiling', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)


    cv2.imshow('Smart Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    