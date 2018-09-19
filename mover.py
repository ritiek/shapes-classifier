import random
import os

def move_all(data_type, shape):
    dirpath = os.path.join(data_type, shape)
    os.makedirs(dirpath, exist_ok=True)
    for filename in os.listdir(shape):
        if filename.endswith('.png'):
            os.rename(os.path.join(shape, filename),
                      os.path.join(data_type, shape, filename))

def move_data(data_type, shape, count):
    dirpath = os.path.join(data_type, shape)
    os.makedirs(dirpath, exist_ok=True)
    for x in random.sample(range(1, 3700), count):
        filename = '{}.png'.format(x)
        os.rename(os.path.join(shape, filename),
                  os.path.join(data_type, shape, filename))


move_data('train', 'circle', 3000)
move_data('train', 'square', 3000)
move_data('train', 'star', 3000)
move_data('train', 'triangle', 3000)

move_all('test', 'circle')
move_all('test', 'square')
move_all('test', 'star')
move_all('test', 'triangle')
