from keras import preprocessing

import numpy as np
import sys

def load_image(image):
    img = preprocessing.image.load_img(image,
                                       color_mode='grayscale',
                                       target_size=(10,10))
    arr_img = preprocessing.image.img_to_array(img)
    return np.expand_dims(arr_img, axis=0)

filenames = sys.argv[1:]
images = []

for filename in filenames:
    image = load_image(filename)
    images.append(image)

images = np.vstack(images)

from keras import models

model = models.load_model('shapes_classifier.h5')
prediction = model.predict(images)

for x in range(len(prediction)):
    for y in range(len(prediction[x])):
        prediction[x][y] = round(prediction[x][y], 3)

print('circle', 'square', 'star', 'triangle')
print(prediction)
