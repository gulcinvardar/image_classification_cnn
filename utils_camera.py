import logging
import pandas as pd
import numpy as np
from  matplotlib import pyplot as plt
import os
import cv2
from tensorflow import keras
from keras.preprocessing import image 
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from utils_model import classes



def write_image(out_folder, frame, filename):
    """
    Writes the image into the test-folder.
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    image_path = f'image-{str(filename)}.png'
    logging.info(f'write image-{filename}')

    cv2.imwrite(os.path.join(out_folder, image_path), frame)
  


def key_action():
    """Taken from course material"""
    # https://www.ascii-code.com/
    k = cv2.waitKey(1)
    if k == 113: # q button
        return 'q'
    if k == 32: # space bar
        return 'space'
    if k == 112: # p key
        return 'p'
    return None


def init_cam(width, height):
    """
    Setups and creates a connection to the webcam.
    Taken from course material.
    """

    logging.info('start web cam')
    cap = cv2.VideoCapture(0)

    # Check success
    if not cap.isOpened():
        raise ConnectionError("Could not open video device")
    
    # Set properties. Each returns === True on success (i.e. correct resolution)
    assert cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    assert cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def read_new_image(out_folder, filename):
    """Reads the image saved by the camera."""
    #time = datetime.now().strftime("%Y%m%d")
    image_path = f'image-{str(filename)}.png'
    img = image.load_img(f'{out_folder}{image_path}', target_size=(128,128))
    img_array = image.img_to_array(img, dtype='uint8')

    return img, img_array

def process_image(img_array):
    """Preprocesses the newly acquired image for the model."""
    proc_img = preprocess_input(img_array)
    proc_img = np.expand_dims(proc_img, axis = 0)/255

    return proc_img

def predict_image(proc_img, m):
    """Predicts the image based on loaded model."""
    pred = m.predict(proc_img)
    max_value = max(pred)
    pred_index = np.argmax(max_value)
    classify = classes[pred_index]

    return classify

def write_on_img(img_array, classify, filename, window_name, out_folder):
    """Writes the prediction on the image and saves it."""
    cv2.putText(
        img= img_array, text=classify, org= (0, 50),
        fontScale=1, color=(255, 0, 0), 
        thickness=1, fontFace=cv2.FONT_HERSHEY_TRIPLEX
        )
    plt.imshow(img_array)
    cv2.imshow(window_name, img_array)
    image.save_img(f'{out_folder}image-{str(filename)}-pred.png', img_array)

    return img_array