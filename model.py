import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,MaxPooling2D,Conv2D,Dropout,Activation,BatchNormalization
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import warnings 

epochs_for_training = 100

class Model:
    def __init__(self):
        self.df = pd.read_csv("./dataset/age_gender.csv")
        self.age_train_data = None
        self.age_test_data = None

        self.gender_train_data = None
        self.gender_test_data = None

        earlystop=EarlyStopping(patience=6)
        learning_rate_reduction=ReduceLROnPlateau(
            monitor='val_acc',
            patience= 3,
            verbose=1,
        )
        self.callbacks = [earlystop, learning_rate_reduction]

        self.age_model = None
        self.gender_model = None

    def init_age_data(self):
        self.num_pixels = len(self.df['pixels'][0].split(" "))
        self.img_height = int(np.sqrt(len(self.df['pixels'][0].split(" "))))
        self.img_width = int(np.sqrt(len(self.df['pixels'][0].split(" "))))
        self.df['pixels'] = self.df['pixels'].apply(lambda x: np.array(x.split(),dtype='float32'))

        X = np.array(self.df['pixels'].tolist())
        X = np.reshape(X, (-1, 48, 48,1))
        y = self.df['age']
        X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X, y, test_size=0.3, random_state=44)
        
        train_datagen=ImageDataGenerator(rescale=1/255)
        train_generator_age=train_datagen.flow(
            X_train_age ,y_train_age ,batch_size=32 
        )

        test_datagen=ImageDataGenerator(rescale=1/255)
        test_generator_age=test_datagen.flow(
            X_test_age ,y_test_age ,batch_size=32 
        )

        self.age_train_data = train_generator_age
        self.age_test_data = test_generator_age

    def init_gender_data(self):
        y = self.df['gender']
        X = np.array(self.df['pixels'].tolist())
        X = np.reshape(X, (-1, 48, 48,1))
        X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(X, y, test_size=0.3, random_state=44)

        train_datagen=ImageDataGenerator(rescale=1/255)
        train_generator_gender =train_datagen.flow(
            X_train_gender ,y_train_gender ,batch_size=32
        )

        test_datagen=ImageDataGenerator(rescale=1/255)
        test_generator_gender =test_datagen.flow(
            X_test_gender ,y_test_gender ,batch_size=32
        )

        self.gender_train_data = train_generator_gender
        self.gender_test_data = test_generator_gender

    def create_age_model(self):
        model_age = Sequential()
        model_age.add(Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)))
        model_age.add(MaxPooling2D(2,2))

        model_age.add(Conv2D(64,(3,3),activation='relu'))
        model_age.add(MaxPooling2D(2,2))

        model_age.add(Conv2D(64,(3,3),activation='relu'))
        model_age.add(MaxPooling2D(2,2))
        model_age.add(Dropout(0.2))

        model_age.add(Conv2D(128,(3,3),activation='relu'))
        model_age.add(MaxPooling2D(2,2))
        model_age.add(Dropout(0.2))          
                
        model_age.add(Flatten())
        model_age.add(Dropout(0.5))            

        model_age.add(Dense(1,activation='relu'))
        model_age.compile(optimizer='adam' ,loss='mean_squared_error',metrics=['mae'])

        self.age_model = model_age

    def create_gender_model(self):
        model_gender = Sequential()
        model_gender.add(Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)))
        model_gender.add(MaxPooling2D(2,2))

        model_gender.add(Conv2D(64,(3,3),activation='relu'))
        model_gender.add(MaxPooling2D(2,2))

        model_gender.add(Conv2D(64,(3,3),activation='relu'))
        model_gender.add(MaxPooling2D(2,2))

        model_gender.add(Flatten())
        model_gender.add(Dense(1,activation='sigmoid'))

        model_gender.compile(optimizer='SGD' ,loss='BinaryCrossentropy',metrics=['accuracy'])

        self.gender_model = model_gender

    def train_age_model(self, epochs: int, ):
        self.age_model.fit(
            self.age_train_data, 
            epochs= epochs,
            validation_data= self.age_test_data,
            callbacks= self.callbacks
        )

    def train_gender_model(self, epochs: int, ):
        self.gender_model.fit(
            self.gender_train_data, 
            epochs= epochs,
            validation_data= self.gender_test_data,
            callbacks= self.callbacks
        )

    def save_age_model(self):
        self.gender_model.save('model/tf_age_model/')

        full_model = tf.function(lambda x: self.age_model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(self.age_model.inputs[0].shape, self.age_model.inputs[0].dtype))

        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir="./frozen_models",
                        name="frozen_age_model.pb",
                        as_text=False)

    def save_gender_model(self):
        self.gender_model.save('model/tf_gender_model/')

        full_model = tf.function(lambda x: self.gender_model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(self.gender_model.inputs[0].shape, self.gender_model.inputs[0].dtype))

        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir="./frozen_models",
                        name="frozen_gender_model.pb",
                        as_text=False)

    def testPlot(self):
        plt.figure(figsize=(20, 20))
        for i in range(25):  
            index = np.random.randint(0, len(self.df))
            plt.xticks([])
            plt.yticks([])   
            plt.grid(False)
            plt.subplot(5, 5, i+1)
            plt.imshow(self.df['pixels'].iloc[index].reshape(48, 48),"gray")
            plt.title(' Age: {}\n Ethnicity: {}\n gender: {}'.format(self.df['age'].iloc[index], {0:"White", 1:"Black", 2:"Asian", 3:"Indian", 4:"Hispanic"}[self.df['ethnicity'].iloc[index]], {0:"Male", 1:"Female"}[self.df['gender'].iloc[index]]),loc="left",color='red',fontsize = 8)

        plt.show()

if __name__ == '__main__':
    model = Model()

    model.init_age_data()
    model.init_gender_data()

    model.create_age_model()
    model.create_gender_model()

    model.train_age_model(epochs_for_training)
    model.train_gender_model(epochs_for_training)

    model.save_age_model()
    model.save_gender_model()
