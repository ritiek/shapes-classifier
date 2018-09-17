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

    directory_labels = { 'circles'   : 0,
                         'squares'   : 1,
                         'stars'     : 2,
                         'triangles' : 3 }

    directory_label = directory_labels[directory]

    labels = np.repeat(directory_label, len(dataset))
    one_hot_labels = preprocessing.utils.to_categorical(labels, len(directory_labels))

    return dataset, list(one_hot_labels)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# TRAIN DATA

circles_train_data, circles_train_labels = load_data('circles', 'train')
squares_train_data, squares_train_labels = load_data('squares', 'train')
stars_train_data, stars_train_labels = load_data('stars', 'train')
triangles_train_data, triangles_train_labels = load_data('triangles', 'train')

train_data = np.array(circles_train_data +
                      squares_train_data +
                      stars_train_data +
                      triangles_train_data)

train_labels = np.array(circles_train_labels +
                        squares_train_labels +
                        stars_train_labels +
                        triangles_train_labels)

del circles_train_data
del circles_train_labels
del squares_train_data
del squares_train_labels
del stars_train_data
del stars_train_labels
del triangles_train_data
del triangles_train_labels

train_data, train_labels = unison_shuffled_copies(train_data, train_labels)

from keras import models
from keras import layers
from keras import regularizers

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(10,10,256)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# VALIDATION DATA

circles_validation_data, circles_validation_labels = load_data('circles', 'validation')
squares_validation_data, squares_validation_labels = load_data('squares', 'validation')
stars_validation_data, stars_validation_labels = load_data('stars', 'validation')
triangles_validation_data, triangles_validation_labels = load_data('triangles', 'validation')

validation_data = np.array(circles_validation_data +
                      squares_validation_data +
                      stars_validation_data +
                      triangles_validation_data)

validation_labels = np.array(circles_validation_labels +
                        squares_validation_labels +
                        stars_validation_labels +
                        triangles_validation_labels)

del circles_validation_data
del circles_validation_labels
del squares_validation_data
del squares_validation_labels

validation_data, validation_labels = unison_shuffled_copies(validation_data, validation_labels)


model.fit(train_data, train_labels, epochs=3, batch_size=128,
          validation_data=(validation_data, validation_labels))

model.save('shapes_classifier.h5')


del train_data
del train_labels
del validation_data
del validation_labels


# TEST DATA

circles_test_data, circles_test_labels = load_data('circles', 'test')
squares_test_data, squares_test_labels = load_data('squares', 'test')
stars_test_data, stars_test_labels = load_data('stars', 'test')
triangles_test_data, triangles_test_labels = load_data('triangles', 'test')

test_data = np.array(circles_test_data +
                     squares_test_data +
                     stars_test_data +
                     triangles_test_data)

test_labels = np.array(circles_test_labels +
                       squares_test_labels +
                       stars_test_labels +
                       triangles_test_labels)

del circles_test_data
del circles_test_labels
del squares_test_data
del squares_test_labels
del stars_test_data
del stars_test_labels
del triangles_test_data
del triangles_test_labels

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
