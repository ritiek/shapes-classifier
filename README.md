# shapes-classifier

A simple shape classifier in Keras using Convolutional 2D networks to classify
shapes from 4 different categories (circles, squares, stars and triangles).


## Usage

### Downloading the code

Clone this repository.

```
$ git clone https://github.com/ritiek/shapes-classifier
$ cd shapes-classifier/
```

### Downloading the dataset

The dataset is available freely on Kaggle - https://www.kaggle.com/smeschke/four-shapes.

Download this dataset, extract `shapes.zip` and place the resultant shape
directories along with the code such that your current directory tree looks like this:

```
.
├── circle
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   └── ...
├── square
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   └── ...
├── star
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   └── ...
├── triangle
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   └── ...
│
├── train.py
├── mover.py
├── predict.py
├── README.md
├── shapes_classifier.h5
```

### Setting up training, validation and test datasets

Once, you've got your directory structured as above. Run
```
$ python mover.py
```
to randomly split 3000 images per shape into `train` and the remaning ~750 images per
shape into `test` directories.

### Training the model

You can now train the model with:
```
$ python train.py
```

For preparing the model, it will use 70% of `train` images for training purposes and
the remaning 30% of `train` images for validation purposes.

It took me about 2 minutes to run this command on CPU on my 4 year-old laptop. This will
also save the resultant model as `shapes_classifier.h5` (There is also a
pre-trained model already included in this repo) and make predictions on `test` images.

### Predicting images

You can load this saved model and make predictions on your images by passing them as
arguments to `predict.py`. For example
```
$ python predict.py 1.png 2.png
```

The model attains an accuracy of ~97% but it doesn't seem to perform well on images outside
the dataset, probably because the images in the dataset are not diverse enough to generalize
better (they comprise of only filled shapes).

## License

`The MIT License`
