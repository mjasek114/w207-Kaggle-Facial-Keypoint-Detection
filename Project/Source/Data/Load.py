import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

from PIL import Image
from PIL import ImageFilter
from math import cos, sin

FTRAIN = '../../Data/training.csv'
FTEST = '../../Data/test.csv'


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        y = y.astype(np.float32) 
        X, y = shuffle(X, y, random_state=42)  # shuffle train data        
    else:
        y = None

    print("X.shape == {0}; X.min == {1:.3f}; X.max == {2:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {0}; y.min == {1:.3f}; y.max == {2:.3f}".format(
        y.shape, y.min(), y.max()))    
    
    return X, y

def transpose_images(X, Y):
    
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]
    
    # Flip 1/2 of the images in this batch at random
    X_flipped = X.copy().reshape(-1,96,96)
    Y_flipped = Y.copy()
    
    bs = X_flipped.shape[0]
    indices = np.random.choice(bs, bs / 10, replace=False)
    
    # simple slice to flip all images
    X_flipped[indices] = X_flipped[indices, :, ::-1]

    
    # Horizontal flip of all x coordinates:
    Y_flipped[indices, ::2] = Y_flipped[indices, ::2] * -1

    # Swap places, e.g. left_eye_center_x -> right_eye_center_x
    for a, b in flip_indices:
        Y_flipped[indices, a], Y_flipped[indices, b] = (
            Y_flipped[indices, b], Y_flipped[indices, a])
    
    return np.vstack((X, X_flipped[indices].reshape(-1,96*96))), np.vstack((Y, Y_flipped[indices]))

def blurr_images(X, Y):

    # Flip 1/2 of the images in this batch at random
    bs = X.shape[0]
    indices = np.random.choice(bs, bs / 10, replace=False)

    X_blurred = X[indices].copy().reshape(-1,96,96)
    X_blurred *= 255.0
    X_blurred = X_blurred.astype(np.ubyte)
    
    for i in range(X_blurred.shape[0]):
        # Blurr the image
        image = Image.fromarray(X_blurred[i], 'L')
        blurred = image.filter(ImageFilter.GaussianBlur)
        
        X_blurred[i] = np.array(blurred.getdata()).reshape(96,96)

    return (X_blurred/255.0).reshape(-1,96*96), Y[indices]

def rotate_images(X, Y):
    
    # Select 1/2 of the images to rotate randomly left or right
    bs = X.shape[0]
    indices = np.random.choice(bs, bs / 10, replace=False)

    X_rot = X[indices].copy().reshape(-1,96,96)
    X_rot *= 255.0
    X_rot = X_rot.astype(np.ubyte)
    
    Y_rot = Y[indices].copy()

    for i in range(X_rot.shape[0]):
        angle = np.random.uniform(-30, 30, 1)
        radians = angle * np.pi / 180

        radians = angle * np.pi / 180
        rotation_matrix = np.array([np.cos(radians), -np.sin(radians), np.sin(radians),  np.cos(radians)])
        rotation_matrix = rotation_matrix.reshape(2,2)

        # Raw points are already centered facepoints around the origin
        # rotate
        rotated_points = np.dot(Y_rot[i].reshape(15,2),rotation_matrix)
        Y_rot[i] = rotated_points.reshape(30)

        # Rotate the image by 'angle'
        image = Image.fromarray(X_rot[i], 'L')
        rot = image.rotate(angle)
        X_rot[i] = np.array(rot.getdata()).reshape(96,96)
        
    return (X_rot/255.0).reshape(-1,96*96), Y_rot  