# Image classification based on a Convolutional Neural Network model

## Results

<p float="left">
  <img src="https://github.com/gulcinvardar/image_classification_cnn/blob/main/example_predicted_images/image-1-pred.png" width="200" />
  <img src="https://github.com/gulcinvardar/image_classification_cnn/blob/main/example_predicted_images/image-2-pred.png" width="200" /> 
  <img src="https://github.com/gulcinvardar/image_classification_cnn/blob/main/example_predicted_images/image-3-pred.png" width="200" />
</p>

## Requirements

Create a virtual Conda environment. 

`conda create -n your_project_name pip python=3.9.12`

`conda activate your_project_name`

`pip install --upgrade pip`

Install:
1. tensorflow
`pip install tensorflow`

2. opencv
`pip install opencv-python`


## Usage
**Step1:**

To capture images to train the model and save them in separate directories, please refer to [github/bonartm](https://github.com/bonartm/imageclassifier)
Take 10-15 images of 10-15 items belonging to one class.
Save the images in different directories named as the class names. 

**Step2:**

Separate random images from each class into another directory names as 'test_data' for further evaluation of the model after training. 
1. run `cnn_model.py`
2. The images are reshaped as 128x128 px.
3. The images are read by ImageDataGenerator imported from image from keras
4. The class names are given by the imagegenerator. See `utils_model.py`
5. Train-validation split is performed by imagegenerator. See `utils_model.py`
6. A multilayer CNN model is built with MaxPool in between layers. 
The Conv2D layers are activated with 'relu' and the dense layer is activated by 'softmax'. Adam is used for optimization. 
The number of neurons in the dense layer is given by traingenerator.
The model is saved as `cnn_objects.h5`

**Step3:**

For further evaluation of the model:

use: `cnn_model_evaluation.ipynb`

**Step4:**

Capture images and predict the class that they belong to. 
1. Create a new directory in the main directory. Call it `data_test_webcam_pred`
2. Go to main directory of the project in your terminal.
3. run:
`python3 capture_and_predict.py`
4. The captured images will be saved and shown again with the predicted class written on it.  




## License

(c) 2022 Gülçin Vardar

Distributed under the conditions of the MIT License. See LICENSE for details.