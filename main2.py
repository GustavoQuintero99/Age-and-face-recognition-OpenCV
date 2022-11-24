import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def showImage(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    #Load image by Opencv2
    img = cv2.imread('./image.jpg', 0)
    input_img = img.astype(np.float32)

    mean = np.array([1.0, 1.0, 1.0]) * 127.5
    scale = 1 / 127.5
    input_blob = cv2.dnn.blobFromImage(
        image=input_img,
        scalefactor=scale,
        size=(48, 48),  # img target size
        mean=mean,
        swapRB=False,  # BGR -> RGB
        crop=True  # center crop
    )

    # Loading model directly from TensorFlow Hub
    cvNet = cv2.dnn.readNetFromTensorflow('./frozen_models/frozen_age_model.pb')  
    #print("OpenCV model was successfully read. Model layers: \n", cvNet.getLayerNames())

    labels = [str(x) for x in range(0,99)]

    cvNet.setInput(input_blob)
    out = cvNet.forward()
    
if __name__ == '__main__':
    main()
