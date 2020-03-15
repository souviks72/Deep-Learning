# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:27:55 2018

@author: souvik
"""

import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

conv_base = VGG16(include_top = False,weights='imagenet',input_shape=(150,150,3))

model = Sequential()
model.add(conv_base)
model.add(Flatten()) ################## VVIP to add this layer
model.add(Dense(256,activation='relu',input_dim=4*4*512))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable == True:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(optimizer=Adam(lr=2e-5),loss='binary_crossentropy',metrics=['acc'])

train_directory = 'cats_dogs/train'
test_directory = 'cats_dogs/test'
val_directory = 'cats_dogs/val'

train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_directory,target_size=(150,150),batch_size=20,class_mode='binary')
val_data = test_datagen.flow_from_directory(val_directory,target_size=(150,150),batch_size=20,class_mode='binary')

history = model.fit_generator(train_data,steps_per_epoch=100,epochs=30,validation_data=val_data,validation_steps=50,)

###PLOTTING RESULTS-------> VERY IMPORTANT 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()