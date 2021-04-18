# Cats vs Dogs
A simple image classification task that predicts whether an image contains a cat or a dog.

## Implementation Details
There are two files - one uses a custom neural net built from scratch, and the other uses a pretrained model available in Keras. Using a pretrained model for your own task is called transfer learning. Both files have three modes - training, testing and none. Model weights are saved to file at the end of each epoch so the model can be loaded at a specific state without training all over again. `cats_and_dogs.py` contains the basic neural network and `cats_and_dogs_transfer.py` contains the transfer learned model. The model to transfer learn is ResNet50 pretrained with weights from the ImageNet challenge. 

This project is meant as a beginner's exercise and impressive results was not the objective. Although I discuss ways to improve performance, there are better alternatives to the tools used here.


## Background

Cats v/s dogs is the "hello-world" of image classification. It is a simple task for humans, but requires training for a machine to achieve. The three phases of the problem are:
  a. Preprocessing
  b. Training
  c. Testing

For preprocessing, we could resize or rescale the images. For data augmentation, we employ techniques like rescaling, zoom-range and horizontal flipping. A generator is created and it sends images batch-wise to your algorithm during training.

The model is trained on a dataset of cats and dogs images. At the end of each epoch model weights are saved to file using callbacks. The fit_generator, used for training images, accepts them in batches as opposed to adding all the images to memory (like while using fit). To re-train the model an older weight file can be loaded and training will continue from there.

For testing, you can create a folder called `testImages` in your working directory. Add photos of cats and dogs (and other similar images) to `testImages`. While testing the model goes through every image file in `testImages` and makes a prediction iteratively.

If you set the mode to none if you only want to check the model summary and do nothing else. 


ResNet was created by Microsoft. The ResNet paper was the first to introduce the concept of residual networks (ResNet). The number that follows (i.e. the 50 in ResNet50) is the depth of the model - the number of layers it has. So, there are many versions of ResNet, with different numbers at the end. ResNet won the ILSVRC competition in 2015 which uses a fraction of the ImageNet dataset for image related tasks. You can find more information about the ResNet model [here](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624)

## Dataset
ImageNet is a dataset containing a few million high resolution images with all sorts of labels. Find out more [here](http://image-net.org/about-overview). 
The dataset I used for training is [here](https://www.kaggle.com/chetankv/dogs-cats-images) (Kaggle). It contains 4000 training images each of cats and dogs; and 1000 testing images of each. Total training sample size is 8000 images and testing sample is 2000 images.

## Results
I ran the first model for 20 epochs and the transfer learned model for 6 epochs. The transfer learned model ran a little slower which could be because of the depth of ResNet and also because of the target dimensions of the images in each file. 
