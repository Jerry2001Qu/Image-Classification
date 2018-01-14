# Image-Classification
An Image Classification Algorithm built with data from the CIFAR-10 dataset.

This program runs a feedforward neural network that received a 55% accuracy rate on a test dataset seperate from the training dataset. The neural network has 3 layers, the first having 1024 nodes as all input images are 32x32, the second having 81 nodes, and the last having 10 nodes corresponding to the 10 different classes.

Prior to processing images through the neural network, all images are reshaped into a grid, transformed into greyscale, and finally reshaped into rows, where each row represents an image in greyscale.

Training the neural network occurs in the main.m file.

Note that this code will not run properly. The dataset files were too large to upload to github but all of the code is here.
