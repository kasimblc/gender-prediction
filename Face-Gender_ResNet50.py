# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:33:00 2020

@author: sumak
"""

from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(ResNet50(include_top=False,pooling="avg",weights="imagenet",input_shape=(128,128,3)))

model.add(Dense(2,activation="softmax"))


model.compile(optimizer="sgd",loss="categorical_crossentropy",metrics=["accuracy"])

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

dataGenerator=ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.1)

trainGenerator=dataGenerator.flow_from_directory("D:\\Mocococo\\B-2\\f-m_images\\",target_size=(128,128),
                                                 batch_size=8,class_mode="categorical",
                                                 subset="training")


valGenerator=dataGenerator.flow_from_directory("D:\\Mocococo\\B-2\\f-m_images\\",target_size=(128,128),
                                                 batch_size=8,class_mode="categorical",
                                                 subset="validation")

fitHistory=model.fit_generator(trainGenerator,
                               steps_per_epoch=trainGenerator.samples//4,
                               epochs=50,
                               validation_data=valGenerator,
                               validation_steps=valGenerator.samples//4)

model.save_weights("Face-Gender-weights.h5")
model.save("Face-Gender-Model-2.h5")