import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(BASE_DIR)
image_dir = os.path.join(BASE_DIR, "images")
# print(image_dir)

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id = current_id + 1
            
            id_ = label_ids[label] 
            # print(label_ids)

            pill_image = Image.open(path).convert("L") #grayscale
            image_array = np.array(pill_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array,1.5,5)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
