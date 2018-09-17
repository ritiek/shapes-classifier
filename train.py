from keras import preprocessing
import numpy as np
import os


def load_data(directory, subdirectory):
    dataset = []
    data = os.path.join(directory, subdirectory)
    for x in os.listdir(data):
        img_path = os.path.join(data, x)
        img = preprocessing.image.load_img(img_path,
                                           color_mode='grayscale',
                                           target_size=(10,10))
        arr_img = preprocessing.image.img_to_array(img)
        one_hot_arr_img = preprocessing.utils.to_categorical(arr_img, 256)
        dataset.append(one_hot_arr_img)

    directory_labels = { 'circle'   : 0,
                         'square'   : 1,
                         'star'     : 2,
                         'triangle' : 3 }

    directory_label = directory_labels[directory]

    labels = np.repeat(directory_label, len(dataset))
    one_hot_labels = preprocessing.utils.to_categorical(labels, len(directory_labels))

    return dataset, list(one_hot_labels)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# TRAIN DATA

circle_train_data, circle_train_labels = load_data('circle', 'train')
square_train_data, square_train_labels = load_data('square', 'train')
star_train_data, star_train_labels = load_data('star', 'train')
triangle_train_data, triangle_train_labels = load_data('triangle', 'train')

train_data = np.array(circle_train_data +
                      square_train_data +
                      star_train_data +
                      triangle_train_data)

train_labels = np.array(circle_train_labels +
                        square_train_labels +
                        star_train_labels +
                        triangle_train_labels)

del circle_train_data
del circle_train_labels
del square_train_data
del square_train_labels
del star_train_data
del star_train_labels
del triangle_train_data
del triangle_train_labels

train_data, train_labels = unison_shuffled_copies(train_data, train_labels)

from keras import models
from keras import layers
from keras import regularizers

model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(10, 10, 256)))
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


# VALIDATION DATA

circle_validation_data, circle_validation_labels = load_data('circle', 'validation')
square_validation_data, square_validation_labels = load_data('square', 'validation')
star_validation_data, star_validation_labels = load_data('star', 'validation')
triangle_validation_data, triangle_validation_labels = load_data('triangle', 'validation')

validation_data = np.array(circle_validation_data +
                      square_validation_data +
                      star_validation_data +
                      triangle_validation_data)

validation_labels = np.array(circle_validation_labels +
                        square_validation_labels +
                        star_validation_labels +
                        triangle_validation_labels)

del circle_validation_data
del circle_validation_labels
del square_validation_data
del square_validation_labels

validation_data, validation_labels = unison_shuffled_copies(validation_data, validation_labels)


model.fit(train_data, train_labels, epochs=3, batch_size=128,
          validation_data=(validation_data, validation_labels))

model.save('shapes_classifier.h5')


del train_data
del train_labels
del validation_data
del validation_labels


# TEST DATA

circle_test_data, circle_test_labels = load_data('circle', 'test')
square_test_data, square_test_labels = load_data('square', 'test')
star_test_data, star_test_labels = load_data('star', 'test')
triangle_test_data, triangle_test_labels = load_data('triangle', 'test')

test_data = np.array(circle_test_data +
                     square_test_data +
                     star_test_data +
                     triangle_test_data)

test_labels = np.array(circle_test_labels +
                       square_test_labels +
                       star_test_labels +
                       triangle_test_labels)

del circle_test_data
del circle_test_labels
del square_test_data
del square_test_labels
del star_test_data
del star_test_labels
del triangle_test_data
del triangle_test_labels

test_data, test_labels = unison_shuffled_copies(test_data, test_labels)

predictions = model.predict(test_data)
labels = test_labels

del test_data
del test_labels

for x in range(len(predictions)):
    for y in range(len(predictions[x])):
        predictions[x][y] = round(predictions[x][y], 3)

print('')
print('Test dataset predictions:')
print(predictions)
print('')
print('Actual test dataset labels:')
print(labels)
