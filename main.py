import tensorflow_hub as hub
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

class CameraTensor:
    def __init__(self, age_model_path, genderModelPath):
        self.age_model_path = age_model_path
        self.gender_model_path = genderModelPath
        self.gender_labels = ['Male', 'Female']

        self.cnn_age_model = None
        self.cnn_gender_model = None

    def load_models(self):
        self.cnn_age_model = cv2.dnn.readNetFromTensorflow(self.age_model_path)  
        self.cnn_gender_model = cv2.dnn.readNetFromTensorflow(self.gender_model_path)

    def get_blob_from_image(self, img):
        scale = 1 / 127.5
        input_blob = cv2.dnn.blobFromImage(
            image=img,
            scalefactor=scale,
            size=(48, 48),
            mean=[0,0,0],
            swapRB=False,
            crop=False
        )
        return input_blob

    def get_age_interval(self, age):
        if age <= 10:
            return "Less than 10"
        elif age >= 11 and age <= 17:
            return "11 to 17"
        elif age >= 18 and age <= 30:
            return "18 to 30"
        elif age >= 31 and age <= 50:
            return "31 to 50"
        elif age >= 51 and age <= 65:
            return "51 to 65"
        else:
            return "More than 65"

    def run(self):
        if self.cnn_gender_model is None or self.cnn_age_model is None:
            raise Exception("Please init the models beofre running the program")
            
        cap = cv2.VideoCapture(0)  
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

                #Save image to further tesitng or model adjustments
                #cv2.imwrite('image2.jpg', faces)

                input_img = faces.astype(np.float32)
                input_blob = self.get_blob_from_image(input_img)

                #Make age predicitions
                self.cnn_age_model.setInput(input_blob)
                age = self.cnn_age_model.forward()
                age = age[0][0] 
                age = round(age)

                #Make gender predictions
                self.cnn_gender_model.setInput(input_blob)
                gender = self.cnn_gender_model.forward()

                #Frame results on camera
                result = f'Age: {self.get_age_interval(age)} gender: {self.gender_labels[round(gender[0][0])]}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(rectangle, result,(x, y - 10), font, 2, (255,0,0), 4, cv2.LINE_AA)

                #Print results on console
                print(f'Current prediction: age({age}) gender({self.gender_labels[round(gender[0][0])]})')
            cv2.imshow('Age and face recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    
def init():
    prog = CameraTensor('./frozen_models/frozen_age_model.pb', './frozen_models/frozen_gender_model.pb')
    prog.load_models()
    prog.run()

if __name__ == '__main__':
    init()



