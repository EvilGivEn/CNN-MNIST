import os
import time
import random
import numpy as np
from keras import backend as K
from keras import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate
from keras.optimizers import rmsprop

from keras.datasets import mnist

num_classes = 10
image_size_height = 28
image_size_width = 28
nb_epoch = 50
batch_size = 1024

path = 'mnist'

nb_train_samples = 60000
nb_validation_samples = 10000

number = [
	np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float), 
	np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float), 
	np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float), 
	np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float), 
	np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float), 
	np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float), 
	np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=float), 
	np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=float), 
	np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=float), 
	np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float), 
	]

def generator(samples, lables, batch_size=128):
	while True:
		samples_size = len(samples)
		times = samples_size // batch_size
		x = [x for x in range(times)]
		random.shuffle(x)
		for i in x:
			yield samples[i * batch_size : (i + 1) * batch_size], lables[i * batch_size : (i + 1) * batch_size]


if K.image_data_format() == 'channels_first':
	input_shape = (1, image_size_height, image_size_width)
else:
	input_shape = (image_size_height, image_size_width, 1)

model_inputs = Input(input_shape)
layer = Conv2D(filters=4, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_normal', input_shape=input_shape)(model_inputs)
layer = Conv2D(filters=16, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_normal')(layer)
layer = Conv2D(filters=32, kernel_size=2, strides=1, activation='relu', kernel_initializer='glorot_normal')(layer)
layer = MaxPooling2D(pool_size=2, padding='same')(layer)
layer = Conv2D(filters=64, kernel_size=2, strides=1, activation='relu', kernel_initializer='glorot_normal')(layer)
average_pooling = AveragePooling2D(pool_size=2, padding='same')(layer)
max_pooling = MaxPooling2D(pool_size=2, padding='same')(layer)
layer = Concatenate()([average_pooling, max_pooling])
layer = Flatten()(layer)
layer = Dense(128, activation='elu')(layer)
layer = Dropout(0.25)(layer)
layer = Dense(128, activation='elu')(layer)
layer = Dropout(0.05)(layer)
layer = Dense(32, activation='elu')(layer)
predictions = Dense(num_classes, activation='softmax')(layer)

model = Model(inputs=model_inputs, outputs=predictions)
model.summary()

opt = rmsprop()

model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

(train_images, train_labels), (validation_images, validation_labels) = mnist.load_data()

train_images = train_images.reshape(60000, 28, 28, 1)

result = np.empty((60000, 10), dtype=float)
pos = 0
for label in train_labels:
	result[pos] = number[label]
	pos += 1
train_labels = result

validation_images = validation_images.reshape(10000, 28, 28, 1)

result = np.empty((10000, 10), dtype=float)
pos = 0
for label in validation_labels:
	result[pos] = number[label]
	pos += 1
validation_labels = result

train_generator = generator(train_images, train_labels, batch_size)
validation_generator = generator(validation_images, validation_labels, batch_size)

history = model.fit_generator(train_generator, 
	steps_per_epoch=(nb_train_samples//batch_size), 
	validation_data=validation_generator,
	validation_steps=(nb_validation_samples//batch_size),
	epochs=nb_epoch)

identation_str = time.strftime("%m%d_%H%M", time.localtime())

model.save('model_{}.hdf5'.format(identation_str))

with open('model_{}_log.txt'.format(identation_str),'w') as f:
	f.write(str(history.history))
