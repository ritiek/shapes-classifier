import os

def move_all(shape, data_type):
    dirpath = os.path.join(shape, data_type)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    for x in os.listdir(shape):
        if x.endswith('.png'):
            os.rename(os.path.join(shape, x), os.path.join(shape, data_type, x))

def move_data(shape, data_type, start, end):
    dirpath = os.path.join(shape, data_type)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    for x in range(start, end):
        filename = '{}.png'.format(x)
        filepath = os.path.join(shape, filename)
        os.rename(filepath, os.path.join(shape, data_type, filename))

move_data('circle', 'train', 0, 3000)
move_data('square', 'train', 0, 3000)
move_data('star', 'train', 0, 3000)
move_data('triangle', 'train', 0, 3000)

move_data('circle', 'validation', 3000, 3500)
move_data('square', 'validation', 3000, 3500)
move_data('star', 'validation', 3000, 3500)
move_data('triangle', 'validation', 3000, 3500)

move_all('circle', 'test')
move_all('square', 'test')
move_all('star', 'test')
move_all('triangle', 'test')
