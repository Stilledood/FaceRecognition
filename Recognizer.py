import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
font= cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
id = 0
names = ['None', 'Florin', 'Trompy']

while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.2, minNeighbors = 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if confidence < 100:
            id = names[id]
            confidence = f"{round(100-confidence)}"
        else:
            id = 'Unknown'
            confidence = f"{round(1---confidence)}"

        cv2.putText(img, str(id),(x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(img,str(confidence),(x+5,y+h-5), font, 1, (255,255,0), 1)
    cv2.imshow('camera',img)
    k = cv2.waitKey(10)
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()