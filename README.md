# Cats vs Dogs
A simple image classification task which predicts whether the image provided is that of a cat or a dog.

# Implementation Details
The code is implemented using the keras library with tensorflow backend in python. 
There are two files - one with its own neural network built from scratch and the other; which uses a pretrained, prebuilt model available in the keras framework. This method of using a pretrained model for your own task is called transfer learning.
Both algorithms have three modes: training, testing and none. After each epoch, the model weights are saved to file so that at any point the model can be loaded at a particular state without training all over again.

The `cats_and_dogs.py` file contains the basic neural network and the `cats_and_dogs_transfer.py` file contains the transfer learning algorithm. I have used the ResNet50 model pretrained with the weights used for the ImageNet challenge. 

# Background

The classic cats v/s dogs problem is the "hello-world" of the image classification task. It is a very simple task in theory and for humans, but requires a considerable amount of training for a machine to achieve. After training we want the machine to be able to take an image as input and predict if it is an image of a cat or a dog. 

The three phases of the problem are:
  a. Preprocessing
  b. training
  c. testing

In the preprocessing phase, we employ several techniques to transform the images - rescaling, zoom-range and horizontally flipping them to name a few. After this data augmentation is done, we must create the generator that will send images batch-wise to your algorithm during training.

In the training phase, the model is trained on the dataset of cats' and dogs' images. At the end of each epoch, the model weights are saved to file using the callbacks feature of keras. We use the fit_generator for training which accepts the images into memory in batches as opposed to bringing all the images into memory at once (by using fit). For re-training, an older weight file is loaded and the training continues from there.

For the testing phase, you will want to go ahead and create a new folder in your working directory called "testImages". Add any photos of cats and dogs (or anything else really) to this new folder. Now in the testing phase we will go through every .jpg file in this directory and make a prediction on each of them iteratively.

You can set the mode to none if you want to neither train nor test the model. This may be helpful if you only want to check the model summary and do nothing else. 

ResNet is a model created by the team at Microsoft. The corresponding paper was the first to introduce the concept of residual networks (ResNet). The number that follows ResNet is usually the depth of the model i.e. the number of layers deep it is, therefore there are several variants of ResNet with different numbers at the end, including ResNet50. ImageNet is a dataset containing a few million high resolution images with all sorts of labels. The ResNet architecture won the ILSVRC competition in 2015 which uses a fraction of the ImageNet dataset for image related tasks. You can find more information about the ResNet model [here](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624) and on the ImageNet challenge [here](http://image-net.org/about-overview).

# Dataset
The dataset I used can be found [here](https://www.kaggle.com/chetankv/dogs-cats-images) (Kaggle). It contains 4000 training images each of cats and dogs; and 1000 testing images of each as well. Total training sample size is 8000 images and for testing is 2000 images.

# Results
I ran the first model for 20 epochs and the transfered learning model for 6 epochs which took me a little under an hour each on my NVIDIA GeForce 940mx GPU (which is not all that powerful). The transfer learning model runs a little slower than the other file. This could be because of the depth of the ResNet model and also due to the target dimensions of the images in the respective files. 

I have included a few results at the top of the cats_and_dogs.py file. Additionally, I have also added all the weight files in .hdf5 format. You might need a library to handle the hdf5 files; I use h5py which can be installed easily using 

`pip install h5py`
