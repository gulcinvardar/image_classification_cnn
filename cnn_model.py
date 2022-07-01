import pandas as pd
import numpy as np
import tensorflow as tf
import cv2

from utils_model import train_generator, validation_generator

from tensorflow import keras
from keras.preprocessing import image 
from keras import backend as K
from keras import layers
from keras.layers import Dense, Input, InputLayer, Flatten
from keras.models import Sequential, Model

from  matplotlib import pyplot as plt



def cnn_model():
    """Creates a cnn model."""
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(len(train_generator.class_indices), activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model


def earlystop():
    """Stops the learning based on val_loss"""
    early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    min_delta=0.0005, 
    patience=3, 
    verbose=1, 
)
    return early_stop

base_path = './data/'
target_size=(128,128)
input_shape = (128, 128, 3)
epochs = 150
batch_size=32

if __name__ == "__main__":

    K.clear_session()
    model = cnn_model()
    early_stop = earlystop()
    model.summary()
    hist = model.fit_generator(train_generator,
                    validation_data = validation_generator, 
                    epochs=epochs, 
                    callbacks = [early_stop]
                    )
    model.save("cnn_objects.h5")
    fig = pd.DataFrame(hist.history).plot()
    plt.savefig('cnn_object_plot.png')
