# Kronos - Deep Learning Helpers and Utilities

## You have just found Kronos
Kronos is a simple and high-level library, written in Python, that provides utilities and helpers to make the first steps working in image recognition as simple and faster as we can.

## Getting started: 30 seconds to Kronos
### Image Preprocessing

`Images` provides a simple chaining API to load and preprocess your input images in a simple and really fast way. For example:

```python
from kronos.preprocessing import Images
# This code performs the following steps:
# - Loads the image file
# - Crop the image discarding the borders and keeping the 95% of the image.
# - Padding the image to square ( pad or crop the image and convert it to a square image without change the ratio)
# - Resize the image to the desired size
# - Convert the image to a numpy array (h,w,c). The matrix is normalized (between 0 and 1) and the mean of each channel is substracted
image_data = Images().load(
    './dataset/images/0001.jpg'
).central_crop(
    central_fraction=0.95
).pad_to_square().resize(
    299, 299
).to_array(
    normalized=True, mean_normalized=([123.68, 116.779, 103.939])
)
```

### Batch Generators
`Generators` provide a unique method to create and process mini-batches from your dataset. For example:

```python
from kronos.training import Generators

X = ['./dog.1.jpg', './dog.2.jpg', './dog.3.jpg', './dog.4.jpg', './cat.1.jpg', './cat.2.jpg', './cat.3.jpg', './cat.4.jpg']
y = ['dog', 'dog', 'dog', 'dog', 'cat', 'cat', 'cat', 'cat']

def print_single_row(X, y):
    print("x:", X, " y:", y)
    return X, y

test = Generators.slice_input_producer(
    X, y, prepare=print_single_row, batch_size=2, shuffle=True
)

for i in range(5):
    print("--"); next(test);
```
```
> --
> x: ./dog.2.jpg  y: dog
> x: ./cat.2.jpg  y: cat
> --
> x: ./cat.1.jpg  y: cat
> x: ./dog.4.jpg  y: dog
> --
> x: ./cat.4.jpg  y: cat
> x: ./cat.3.jpg  y: cat
> --
> x: ./dog.1.jpg  y: dog
> x: ./dog.3.jpg  y: dog
> --
> x: ./dog.2.jpg  y: dog
> x: ./cat.2.jpg  y: cat
> --
```

## Simple example in Keras
```python
from kronos.preprocessing import Images
from kronos.training import Generators

X = [...] # An array containing all the image urls
y = [...] # An array containing all the image classes

# ...

def encode_image(X, y):
    # X = image url like "./dog.001.jpg"
    # y = class like "dog" or "1"
    # Return a 224x224 normalized squared image and its class
    return Images().load(X).resize(224, 224).to_array(normalized=True), y

#...

model.fit_generator(
    Generators.slice_input_producer(X_train, y_train, prepare=encode_image, batch_size=32, shuffle=True),
    samples_per_epoch=256,
    nb_epoch=1000,
    validation_data=Generators.slice_input_producer(X_test, y_test, prepare=encode_image, batch_size=32, shuffle=True),
    nb_val_samples=128
)
```
