from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
import os 
from glob import glob
import os 
import cv2
import matplotlib.pyplot as plt


class Model:
    def __init__(self,train_path,test_path):
        self.train_path = train_path
        self.test_path = test_path 
        self.model = self.model()

    def model(self,):
        model = Sequential()
        model.add(Conv2D(filters = 32,kernel_size = (3,3),activation='relu',padding = 'same',input_shape = (150,150,3))) #150,150,32
        model.add(Conv2D(filters = 64,kernel_size = (3,3),activation='relu')) #148,148,64
        model.add(MaxPooling2D(pool_size=(2,2))) #74,74,64
        model.add(Conv2D(filters = 128,kernel_size = (3,3),activation='relu')) #72,72,128
        model.add(MaxPooling2D(pool_size=(2,2))) #36,36,128
        model.add(Conv2D(filters = 128,kernel_size = (5,5),activation='relu')) #32,32,128
        model.add(Flatten())
        model.add(Dense(3,activation='softmax')) 

        model.compile(optimizer=Adam(),loss = 'categorical_crossentropy',metrics=['accuracy'])

        return model 
    
    def data_transformation(self,train_path,test_path):
        train_path = self.train_path
        test_path = self.test_path
        train_datagen = ImageDataGenerator(rescale=1./225)
        test_datagen = ImageDataGenerator(rescale = 1./255)
        train_dir = train_datagen.flow_from_directory(train_path,
                                              target_size=(150,150),
                                              batch_size = 10,
                                              class_mode= 'categorical')

        test_dir = test_datagen.flow_from_directory(test_path,
                                                    target_size=(150,150),
                                                    batch_size = 10,
                                                    class_mode= 'categorical')
        
        return train_dir , test_dir 

    def inisate_training(self):
        model = self.model
        train_dir , test_dir = self.data_transformation(self.train_path, self.test_path)
        r = model.fit(
        train_dir,
        validation_data = test_dir,
        epochs=10,
        steps_per_epoch=len(train_dir),
        validation_steps=len(test_dir)
        )

        model_save_path = os.path.join(os.curdir,'model.h5')

        model.save(model_save_path)

        return model_save_path 
        

