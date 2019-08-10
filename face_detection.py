import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    orig_labels = pickle.load(f)
    labels = {v:k for k,v in orig_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        id_,conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            # print(id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        eyes = eye_cascade.detectMultiScale(roi_gray)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()