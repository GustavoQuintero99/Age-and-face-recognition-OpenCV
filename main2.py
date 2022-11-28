import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def showImage(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getBlobFromImage(img):
    scale = 1 / 127.5
    input_blob = cv2.dnn.blobFromImage(
        image=img,
        scalefactor=scale,
        size=(48, 48),  # img target size
        mean=[104,117,123],
        swapRB=False,  # BGR -> RGB
        crop=False  # center crop
    )
    return input_blob

def test():
    df = pd.read_csv("./dataset/age_gender.csv")
    num_pixels = len(df['pixels'][0].split(" "))
    img_height = int(np.sqrt(len(df['pixels'][0].split(" "))))
    img_width = int(np.sqrt(len(df['pixels'][0].split(" "))))
    df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(),dtype='float32'))

    print(f'img height: {img_height}, img width: {img_width}')
    print(f'num pixels: {num_pixels}')

    cvNet = cv2.dnn.readNetFromTensorflow('./frozen_models/frozen_age_model.pb') 
    cvNet2 = cv2.dnn.readNetFromTensorflow('./frozen_models/frozen_gender_model.pb')   
    plt.figure(figsize=(20, 20))
    for i in range(25):  
        index = np.random.randint(0, len(df))
        img = df['pixels'].iloc[index].reshape(48, 48)
        print(img)
        img = getBlobFromImage(img)
        cvNet.setInput(img)
        cvNet2.setInput(img)
        out = cvNet.forward()
        gender = cvNet2.forward()
        plt.xticks([])
        plt.yticks([])   
        plt.grid(False)
        plt.subplot(5, 5, i+1)
        plt.imshow(df['pixels'].iloc[index].reshape(48, 48),"gray")
        plt.title(' Age: {}\n Gender: {}\n Predicted age: {}\n Predicted gender: {}'.format(df['age'].iloc[index], {0:"Male", 1:"Female"}[df['gender'].iloc[index]], round(out[0][0]), "Female" if round(gender[0][0]) == 1 else "Male"), loc="left",color='red',fontsize = 8)

    plt.show()

def main():
    #Load image by Opencv2
    img = cv2.imread('./image2.jpg', 0)
    #input_img = img.astype(np.float32)

    scale = 1 / 127.5
    input_blob = cv2.dnn.blobFromImage(
        image=img,
        scalefactor=scale,
        size=(48, 48),  # img target size
        mean=[104,117,123],
        swapRB=False,  # BGR -> RGB
        crop=False  # center crop
    )

    cv2.waitKey(5000)
    # Loading model directly from TensorFlow Hub
    cvNet = cv2.dnn.readNetFromTensorflow('./frozen_models/frozen_age_model.pb')  
    #print("OpenCV model was successfully read. Model layers: \n", cvNet.getLayerNames())

    labels = [str(x) for x in range(0,99)]

    cvNet.setInput(input_blob)
    out = cvNet.forward()
    print(out)
    
if __name__ == '__main__':
    test()
