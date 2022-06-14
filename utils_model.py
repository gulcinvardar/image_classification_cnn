# %%
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import cv2
import matplotlib.image as mpimg
from tensorflow import keras
from keras.preprocessing import image 
from tensorflow.keras.applications.resnet50 import preprocess_input
from  matplotlib import pyplot as plt


def class_names(train_datagen):
    """Retrieves the class names based on the folder names"""
    all_data = train_datagen.flow_from_directory(
    directory=base_path,
    class_mode="input",
    )
    classes = list(all_data.class_indices.keys())

    return classes

def image_generator():
    """Opens and preprocesses the data-set"""
    train_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1.0/255.0,
    validation_split=0.2,
    )

    return train_datagen


def train_val_sets(train_datagen):
    """Creates the train and test data-sets"""
    train_generator = train_datagen.flow_from_directory(
    directory=base_path,
    target_size=target_size, 
    classes = classes,
    class_mode="categorical",
    batch_size=batch_size,
    subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
    directory=base_path,
    target_size=target_size, 
    classes = classes,
    class_mode="categorical",
    batch_size=batch_size,
    subset='validation')
    
    return train_generator, validation_generator

def images_for_pred():
    """To predict the images in the test-folder"""
    pred_batches = image.ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    rescale=1.0/255.0
    ).flow_from_directory(
    directory=test_path, 
    target_size=target_size, 
    classes=['test'],
    shuffle=False)

    return pred_batches

base_path = './data/'
test_path ='./test_data/'
target_size=(128,128)
input_shape = (128, 128, 3)
epochs = 150
batch_size=32

train_datagen = image_generator()
classes = class_names(train_datagen)
train_generator, validation_generator = train_val_sets(train_datagen)

