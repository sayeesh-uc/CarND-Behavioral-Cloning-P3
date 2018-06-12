# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center_2018_06_11_16_55_56_353.jpg "Center lane driving"
[image3]: ./examples/center_2018_06_12_11_50_27_781.jpg "Recovery Image"
[image5]: ./examples/center_2018_06_11_19_22_31_834.jpg "Recovery Image"
[image6]: ./examples/center_2018_06_11_16_55_56_353.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of:

| Layer                 |     Description                       |
|-----------------------|:-------------------------------|
| Input                 | 160x320x3 image                       |
| Normalization         | pixel value/255.0 - 0.5      |
| Cropping              | Crop top 50 and bottom 20 pixels|
| Convolution 5x5       | 5x5 kernel, 24 outputs|
| Pooling 2x2           | 2x2 pool_size|
| RELU                  |         |                                              |
| Convolution 5x5       | 5x5 kernel, 36 outputs|
| Pooling 2x2           | 2x2 pool_size|
| RELU                  |         |  
| Convolution 5x5       | 5x5 kernel, 48 outputs|
| Pooling 2x2           | 2x2 pool_size|
| RELU                  |         |  
| Dropout               | ratio of 0.25 dropout to not overfit|
| Convolution 3x3       | 3x3 kernel, 64 outputs|
| Pooling 2x2           | 2x2 pool_size|
| RELU                  |         |
| Convolution 3x3       | 3x3 kernel, 64 outputs|
| Pooling 2x2           | 2x2 pool_size|
| RELU                  |         |
| Flatten               |     |
| Fully connected       | output 100        |
| Dropout               | ratio of 0.25 dropout to not overfit|
| Fully connected       | output 50          |
| Fully connected       | output 10           |
| Fully connected       | output 1 (measurement)   |


The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the road. I see one instance it goes off track. Adding more data will solve the problem.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep the car on the road while driving.

My first step was to use a convolution neural network model similar to the Nvidia network as suggested. I thought this model might be appropriate because it was designed for self drving which can handle corrections from left and right images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropouts.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, by taking the car outside of the track and then start recording.

There were few spots in the road I had to work a lot on getting corse recovery data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

The tough challenges are:
1. collecting data
2. collecting "recover back to center" data
3. diffuculty in removing bad data
4. ability to get good driving data when we run the car in simulator mode.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 9000+ number of data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by increase in validation effor after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
