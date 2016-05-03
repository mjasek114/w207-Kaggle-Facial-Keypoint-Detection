# W207 Final Project - Kaggle Facial Keypoints Detection - Spring 2016

## Team
Megan Jasek, Charles Kekeh, James King and Beth Partridge

## Description
The object of the competition was to detect 15 locations on the human face (keypoints) given a digital image.  The images were 96x96 pixels.  Each keypoint was designated by an x and y coordinate.  A set of labeled images was provided for training a model.  Not all training images had all keypoints labeled.

This type of problem is a key building block for many applications and is very challenging for the following reasons:
* Feature variation person to person
* 3D pose
* Position
* Viewing angle
* Illumination conditions

For this particular problem there was not enough data to create a model that was not overfit.  There were only 2,000 completely labeled images.  In a convolutional neural network there could be 10,000 or more features presented to the fully connected neural network in the final layer.  With potentially 10,000 features and only 2,000 complete training images, there will be a tendency to overfit the model.

## Solution
Theano, Lasange and AWS were used to implement a solution.  In the AWS environment a g2.2xlarge server that contained a GPU was used to test solutions.  The final solution consisted of a convolutional neural network that was structured with an architecture like the following:  32 * 64 * 128 feature maps, 3 * 3 patch widths and (2, 2) max pooling: 128 feature frames of size 10 *
10 pixels on the last feature map layer.
