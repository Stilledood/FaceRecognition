import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
video = cv2.VideoCapture(0)
while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors =5)
    for (x,y, w , h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0),2)
    cv2.imshow('image',frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()