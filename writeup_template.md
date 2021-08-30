# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is the one proposed by nvidia for end to end self driving cars. It can be found in this link https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf 
It consists of the following layers : (model.py lines 116-134) 
1) Lambda normalisation layer to normalise the inputs
2) 3 5x5 convolution layers of same padding with increasing depth 24, 36, 48.
3) 2 3x3 convolution layers of same padding with same depth of 64.
4) Flattening layer to flatten the input image.
5) Three fully connected layers of 100, 50, 10 neurons in each layer.
The model includes RELU activation to introduce nonlinearity in convolutional layers and dropouts between fully connected layers, and the data is normalized in the model using a Keras lambda layer (code line 118). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 25). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 136).

#### 4. Appropriate training data

I used the default data provided in the project to train the first draft. Then I collected further data to perfect the model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used the model architecture from nvidia's end to end self driving publication. It used 3 5x5 convolutional layer and 2 3x3 convolutional layer. It follows it with 3 fully connected layers. I introdcued RELU activation in convolutional layer to introduce non linearity. 
I also used dropouts between fully connected layers to reduce overfitting. 

#### 2. Final Model Architecture

The final model architecture (model.py lines 116-134) consisted of a convolution neural network with the following layers and layer sizes :

5x5x24 with same padding and ReLU activation.
5x5x36 with same padding and ReLU activation.
5x5x48 with same padding and ReLU activation.

3x3x64 with same padding and ReLU activation.
3x3x64 with same padding and ReLU activation.

A Fully connected layer of 100 neurons with tanh activation.
A Fully connected layer of 50 neurons with tanh activation.
A Fully connected layer of 10 neurons with tanh activation.
A Fully connected layer of 1 neuron with tanh activation.

Here is a visualization of the architecture

![nVidia_model](examples/nVidia_model.png)

#### 3. Creation of the Training Set & Training Process

I used the default data provided in the project to train the first draft. The images from center camera are flipped and images from left and right camera are adjusted with a factor of 0.2 to their steering angle. The images form left and right camera are also flipped to augment the data and improve the number of available data.
Then I collected further data to perfect the model. Model tuning is done in the script model_pre_trained.py.

model_pre_trained.py uses the previously trained model and retrains it on a different data set starting with previously trained weights. Datasets in folders data2 and data4 contains the segment of the track for which initial model fails to perform well. They also contain recovery datasets to recover from undesirable behavious identified in the first draft. This can be seen in the video when the car makes a recovery turn while trying to steer further left and hit the ledge.

