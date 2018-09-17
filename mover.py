import os

def move_all(shape, data_type):
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
move_data('squares', 'train', 0, 3000)
move_data('stars', 'train', 0, 3000)
move_data('triangles', 'train', 0, 3000)

move_data('circle', 'validation', 3000, 3500)
move_data('squares', 'validation', 3000, 3500)
move_data('stars', 'validation', 3000, 3500)
move_data('triangles', 'validation', 3000, 3500)

move_all('circle', 'test')
move_all('squares', 'test')
move_all('stars', 'test')
move_all('triangles', 'test')
