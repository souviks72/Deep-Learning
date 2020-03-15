# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:48:14 2018

@author: souvik
"""

from keras import layers
from keras.models import Sequential
classifier = Sequential()
classifier.add(layers.Conv2D(32,(3,3),input_shape = (150,150,3),activation="relu"))
classifier.add(layers.MaxPool2D(2,2))
classifier.add(layers.Conv2D(64,(3,3),activation="relu"))
classifier.add(layers.MaxPool2D(2,2))
classifier.add(layers.Conv2D(128,(3,3),activation="relu"))
classifier.add(layers.MaxPool2D(2,2))
classifier.add(layers.Conv2D(256,(3,3),activation="relu"))
classifier.add(layers.MaxPool2D(2,2))
classifier.add(layers.Flatten())
classifier.add(layers.Dropout(0.5))
classifier.add(layers.Dense(512,activation="relu"))
classifier.add(layers.Dense(1,activation="sigmoid"))
classifier.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

from keras.preprocessing import image

train_data_generator = image.ImageDataGenerator(
                        rescale=1./255,
                        shear_range=0.4,
                        rotation_range=0.4,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True)

test_data_generator = image.ImageDataGenerator(rescale=1./255)
train_generator = train_data_generator.flow_from_directory(
                    "cats_dogs/train",
                    target_size=(150,150),
                    batch_size=32,
                    class_mode='binary')

val_generator = test_data_generator.flow_from_directory(
                    "cats_dogs/val",
                    target_size=(150,150),
                    batch_size=32,
                    class_mode='binary')

history = classifier.fit_generator(
            train_generator,
            steps_per_epoch=8000,
            epochs=6,
            validation_data=val_generator,
            validation_steps=2000,)


print(classifier.summary)
print(train_generator.class_indices)
import numpy as np
test_image = image.load_img('single_prediction/dog.4049.jpg', target_size = (150,150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result == 1:
    print('dog')
else:
    print('cat')
###FINE TILL 4TH EPOCH
