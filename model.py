import csv
import cv2
import numpy as np

# Read the driving log file.
lines = []
with open('./drive/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# Split the data into training and validation set.
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print("length of train samples: %d" % len(train_samples))


import sklearn

# Generator function to return a batch of data at a time.
def generator(samples, batch_size=32, message=""):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []

            for line in batch_samples:
                paths = line[0:3]
                corrections = [0, 0.2, -0.2]
                for source_path, correction in zip(paths, corrections):
                    file_name = source_path.split('/')[-1]
                    current_path = './drive/data/IMG/' + file_name
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(line[3]) + correction
                    measurements.append(measurement)

                    image_flipped = np.fliplr(image)
                    measurement_flipped = -measurement
                    images.append(image_flipped)
                    measurements.append(measurement_flipped)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

            
train_generator = generator(train_samples, batch_size=32, message="train_data")
validation_generator = generator(validation_samples, batch_size=32, message="validation_data")
           

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

# Model from Nvidia.

model = Sequential()
# Normalization layer.
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))

# Crop less valuable information in the image.
model.add(Cropping2D(cropping=((50,20), (0,0))))


model.add(Convolution2D(24, kernel_size=(5,5)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(36, kernel_size=(5,5)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(48, kernel_size=(5,5)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Convolution2D(64, kernel_size=(3,3)))
model.add(Activation('relu'))

model.add(Convolution2D(64, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.25))

model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Use mse as loss function as this is regression.
model.compile(loss='mse', optimizer ='adam')

model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/32,
validation_data=validation_generator, validation_steps=len(validation_samples)/32, epochs=5, verbose = 2)

model.save('model.h5')