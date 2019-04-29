# Cats vs Dogs
A simple image classification task which predicts whether the image provided is that of a cat or a dog.

# Implementation Details
The code is implemented using the keras library with tensorflow backend in python. 
The algorithm has three modes: training, testing and none. After each epoch, the model weights are saved to file so that at any point the model can be loaded at a particular state without training all over again.

# Background

The classic cats v/s dogs problem is the "hello-world" of the image classification task. It is a very simple task in theory and for humans, but requires a considerable amount of training for a machine to achieve. After training we want the machine to be able to take an image as input and predict if it is an image of a cat or a dog. 

The three phases of the problem are:
  a. Preprocessing
  b. training
  c. testing

In the preprocessing phase, we employ several techniques to transform the images - rescaling, zoom-range and horizontally flipping them to name a few. After this data augmentation is done, we must create the generator that will send images batch-wise to your algorithm during training.

In the training phase, the model is trained on the dataset of cats' and dogs' images. At the end of each epoch, the model weights are saved to file using the callbacks feature of keras. We use the fit_generator for training which accepts the images into memory in batches as opposed to bringing all the images into memory at once (by using fit). 

For the testing phase, you will want to go ahead and create a new folder in your working directory called "testImages". Add any photos of cats and dogs (or anything else really) to this new folder. Now in the testing phase we will go through every .jpg file in this directory and make a prediction on each of them iteratively.

You can set the mode to none if you want to neither train nor test the model. This may be helpful if you only want to check the model summary and do nothing else. 

# Dataset
The dataset I used can be found [here](https://www.kaggle.com/chetankv/dogs-cats-images) (Kaggle). It contains 4000 training images each of cats and dogs; and 1000 testing images of each as well. 

# Results
I ran the model for only 5 epochs which took me a little under 2 hours on my NVIDIA GeForce 940mx GPU (which is not all that powerful so if you have a GPU don't worry its manageable). I have included my exact results at the top of the cats_and_dogs.py file. Additionally, I have also added all the weight files in .hdf5 format. You might need a library to handle the .hdf5 files; I use h5py which can be installed easily using 

`pip install h5py`

# Notes
I plan on adding a retraining mode to the model soon, where you can load an older weight_file and continue training from there. It's nothing special, just a few extra lines to the training code block. 
