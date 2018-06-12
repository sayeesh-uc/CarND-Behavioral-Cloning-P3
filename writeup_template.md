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

|Layer (type)                                     |                  Output Shape                                     |           Param #                |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
|lambda_1 (Lambda)                                     |             (None, 160, 320, 3)                                     |    0                      |
|cropping2d_1 (Cropping2D)                                     |     (None, 90, 320, 3)                                     |     0                      |
|conv2d_1 (Conv2D)                                     |             (None, 86, 316, 24)                                     |    1824                   |
|max_pooling2d_1 (MaxPooling2D)                                     |(None, 43, 158, 24)                                     |    0                      |
|activation_1 (Activation)                                     |     (None, 43, 158, 24)                                     |    0                      |
|conv2d_2 (Conv2D)                                     |             (None, 39, 154, 36)                                     |    21636                  |
|max_pooling2d_2 (MaxPooling2D)                                     |(None, 19, 77, 36)                                     |     0                      |
|activation_2 (Activation)                                     |     (None, 19, 77, 36)                                     |     0                      |
|conv2d_3 (Conv2D)                                     |             (None, 15, 73, 48)                                     |     43248                  |
|max_pooling2d_3 (MaxPooling2D)                                     |(None, 7, 36, 48)                                     |      0                      |
|activation_3 (Activation)                                     |     (None, 7, 36, 48)                                     |      0                      |
|dropout_1 (Dropout)                                     |           (None, 7, 36, 48)                                     |      0                      |
|conv2d_4 (Conv2D)                                     |             (None, 5, 34, 64)                                     |      27712                  |
|activation_4 (Activation)                                     |     (None, 5, 34, 64)                                     |      0                      |
|dropout_2 (Dropout)                                     |           (None, 5, 34, 64)                                     |      0                      |
|conv2d_5 (Conv2D)                                     |             (None, 3, 32, 64)                                     |      36928                  |
|activation_5 (Activation)                                     |     (None, 3, 32, 64)                                     |      0                      |
|dropout_3 (Dropout)                                     |           (None, 3, 32, 64)                                     |      0                      |
|flatten_1 (Flatten)                                     |           (None, 6144)                                     |           0                      |
|dense_1 (Dense)                                     |               (None, 100)                                     |            614500                 |
|activation_6 (Activation)                                     |     (None, 100)                                     |            0                      |
|dropout_4 (Dropout)                                     |           (None, 100)                                     |            0                      |
|dense_2 (Dense)                                     |               (None, 50)                                     |             5050                   |
|activation_7 (Activation)                                     |     (None, 50)                                     |             0                      |
|dropout_5 (Dropout)                                     |           (None, 50)                                     |             0                      |
|dense_3 (Dense)                                     |               (None, 10)                                     |             510                    |
|activation_8 (Activation)                                     |     (None, 10)                                     |             0                      |
|dense_4 (Dense)                                     |               (None, 1)                                     |              11                     |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
|Total params: 751,419|
|Trainable params: 751,419|
|Non-trainable params: 0|
|


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
