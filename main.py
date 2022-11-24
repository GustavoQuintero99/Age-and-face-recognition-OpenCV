import tensorflow_hub as hub
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras

ageNet = cv2.dnn.readNetFromTensorflow('./frozen_models/frozen_age_model.pb')  
genderNet = cv2.dnn.readNetFromTensorflow('./frozen_models/frozen_gender_model.pb')

cap = cv2.VideoCapture(0)

width = 512
height = 512


# [10-, 10-18, 19-30, 31-60, 60+]
def getInterval(age):
    if age <= 10:
        return "-10"
    elif age >= 11 and age <= 18:
        return "10 - 18"
    elif age >= 19 and age <= 30:
        return "19 - 30"
    elif age >= 31 and age <= 60:
        return "30 - 60"
    else:
        return "+60"

genderLabels = ['Female', 'Male']
while(True):
    ret, frame = cap.read()
    
    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_alt2.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), 
                    (0, 0, 255), 2)
        faces = frame[y:y + h, x:x + w]

        faces = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
        input_img = faces.astype(np.float32)
        mean = np.array([1.0, 1.0, 1.0])
        scale = 1
        input_blob = cv2.dnn.blobFromImage(
            image=input_img,
            scalefactor=scale,
            size=(48, 48),  # img target size
            mean=mean,
            swapRB=False,  # BGR -> RGB
            crop=False  # center crop
        )
        ageNet.setInput(input_blob)
        age = ageNet.forward()
        age = age[0][0] / 100 
        age = getInterval(round(age)) 

        genderNet.setInput(input_blob)
        gender = genderNet.forward()
        result = f'Age: {age} gender: {genderLabels[round(gender[0][0])]}'

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(rectangle, result,(x, y - 10), font, 3, (255,0,0), 5, cv2.LINE_AA)
    
    cv2.imshow('black and white', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()