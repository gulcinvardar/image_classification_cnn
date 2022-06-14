import logging
import os
import cv2
from  matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from utils_camera import write_image, key_action, init_cam, read_new_image, process_image, predict_image, write_on_img
from utils_model import classes



if __name__ == "__main__":

    ###This is based on "https://github.com/bonartm/imageclassifier". 
    # It is modified to predict the captured image and retrieve it back with the result written on it.

    
    filename = 1
    out_folder = './data_test_webcam/'
    out_folder_pred = './data_test_webcam_pred/'
    m = load_model("cnn_objects.h5")
    window_name = 'Image'

    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    logging.getLogger().setLevel(logging.INFO)
   
    # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None
    
    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            # draw a [224x224] rectangle into the frame, leave some space for the black border 
            offset = 2
            width = 224
            x = 160
            y = 120
            cv2.rectangle(img=frame, 
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 0, 0), 
                          thickness=2
            )     
            
            # get key event
            key = key_action()
        
            if key == 'space':
                # write the image without overlay
                # extract the [224x224] rectangle out of it
                # predict the image based on model
                # return the captured image and its prediction, and save it
                image = frame[y:y+width, x:x+width, :]
                write_image(out_folder, image, filename)
                img, img_array =  read_new_image(out_folder, filename)
                proc_img = process_image(img_array)
                classify = predict_image(proc_img, m)
                write_on_img(img_array, classify, filename, window_name, out_folder_pred)
                plt.imshow(img_array)
                cv2.imshow(window_name, img_array)
                filename = filename+1 
                


            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)            
            
    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows() 

