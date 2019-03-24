#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:47:39 2019

@author: ovishake
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from glob import glob
import os
import keras

from keras import backend as K


xray_data = pd.read_csv('train.csv')
my_glob = glob('images/*.png')
full_img_paths = {os.path.basename(x): x for x in my_glob}
xray_data['full_path'] = xray_data['Image Index'].map(full_img_paths.get)
print(len(xray_data['full_path']))
xray_data['Finding_strings']=xray_data['Finding Labels'].map(lambda x: 'Pneu' if 'Pneumothorax' in x else 'NoPn')

xray_data_test = pd.read_csv('test.csv')
xray_data_test['full_path'] = xray_data_test['Image Index'].map(full_img_paths.get)
print(len(xray_data_test['full_path']))
xray_data_test['Finding_strings']=xray_data_test['Finding Labels'].map(lambda x: 'Pneu' if 'Pneumothorax' in x else 'NoPn')

xray_data_valid = pd.read_csv('valid.csv')
xray_data_valid['full_path'] = xray_data_valid['Image Index'].map(full_img_paths.get)
print(len(xray_data_valid['full_path']))
xray_data_valid['Finding_strings']=xray_data_valid['Finding Labels'].map(lambda x: 'Pneu' if 'Pneumothorax' in x else 'NoPn')



dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

from keras.preprocessing.image import ImageDataGenerator
data_gen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

image_size = (224, 224)
#data from the df xray_data
train_gen = data_gen.flow_from_dataframe(
        dataframe= xray_data,
        directory=None,
        x_col='full_path',
        y_col='Finding_strings',
        target_size=image_size,
        color_mode= 'rgb',
        class_mode="categorical",
        shuffle=True,
        batch_size = 16)
#data from the df xray_data_test
test_gen = data_gen.flow_from_dataframe(
        dataframe = xray_data_test,
        directory=None,
        x_col='full_path',
        y_col='Finding_strings',
        target_size=image_size,
        color_mode='rgb',
        class_mode="categorical",
        shuffle=True,
        batch_size=16)

step_size_train= train_gen.n // train_gen.batch_size
step_size_valid = test_gen.n // test_gen.batch_size
test_x, test_y = next(data_gen.flow_from_dataframe(
        dataframe = xray_data_valid,
        directory=None,
        x_col='full_path',
        y_col='Finding_strings',
        target_size=image_size,
        color_mode='rgb',
        class_mode="categorical",
        shuffle=True,
        batch_size=16))


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential


bm = keras.applications.DenseNet121(
        include_top=False,
        input_shape=(224,224,3),
        pooling='max')

x = bm.output
pred= Dense(2,activation='sigmoid')(x)

model = keras.models.Model(inputs=bm.input, output=pred)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='best_Densenet_weights.hdf5', verbose=1, save_best_only = True)
callbacks_list = [checkpointer]

model.fit_generator(generator = train_gen, 
        steps_per_epoch = step_size_train, 
        epochs = 1, 
        callbacks = callbacks_list, 
        validation_data = (test_x, test_y),
        shuffle=True,
        validation_steps=step_size_valid)
