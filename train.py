from keras import preprocessing
import numpy as np
import os

# TRAIN & VALIDATION DATA

train_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                       rotation_range=40,
                                                       width_shift_range=0.2,
                                                       height_shift_range=0.2,
                                                       shear_range=0.2,
                                                       zoom_range=0.2,
                                                       horizontal_flip=True,
                                                       fill_mode='nearest',
                                                       validation_split=0.3)

print('Preparing training data')
train_generator = train_datagen.flow_from_directory('train/',
                                                    color_mode='grayscale',
                                                    target_size=(10,10),
                                                    subset='training')

print('Preparing validation data')
validation_generator = train_datagen.flow_from_directory('train',
                                                    color_mode='grayscale',
                                                    target_size=(10,10),
                                                    subset='validation')

from keras import models
from keras import layers
from keras import regularizers

model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(10, 10, 1)))
model.add(layers.MaxPooling2D((1, 1)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((1, 1)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit_generator(train_generator, epochs=10,
                    validation_data=validation_generator)

model.save('shapes_classifier.h5')

# TEST DATA

print('')
print('Preparing test data for making predictions')
test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = train_datagen.flow_from_directory('test/',
                                                   color_mode='grayscale',
                                                   target_size=(10,10),
                                                   shuffle=False)

print('The probablility of shape is depicted as {}'.format(
                        list(test_generator.class_indices.keys())))

predictions = model.predict_generator(test_generator)
print('')

for x in range(len(predictions)):
    rounded_predictions = []
    for y in range(len(predictions[x])):
        rounded_predictions.append(round(predictions[x][y], 3))

    values = { 'prediction' : rounded_predictions,
               'filename'   : test_generator.filenames[x] }
    print(values)
