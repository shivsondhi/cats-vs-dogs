'''
Cats and Dogs - Image Classification using ResNet50 Transfer Learning.
4000 training images of cats and dogs each + 1000 test images each.

TRIAL 1 : 	125 steps/epoch + batch_size of 32 	|| 	binary_crossentropy + sigmoid, validation_steps = 32 with batch_size as well
>			epoch = 1;		loss 	 = 0.1400;		acc 	= 0.9475
>			time: 147s; 	val_loss = 0.0704;		val_acc = 0.9727
>
>			epoch = 2;		loss 	 = 0.0920;		acc 	= 0.9652
> 			time: 142s;		val_loss = 0.0791; 		val_acc = 0.9692
>
>			eopch = 3;		loss  	 = 0.0681;		acc 	= 0.9740
>			time: 144s;		val_loss = 0.0744;		val_acc = 0.9727
>
>			eopch = 4;		loss  	 = 0.0787;		acc 	= 0.9695
>			time: 147s;		val_loss = 0.0577;		val_acc = 0.9844
>
>			eopch = 5;		loss  	 = 0.0695;		acc 	= 0.9702
>			time: 141s;		val_loss = 0.0508;		val_acc = 0.9792
>
>			eopch = 6;		loss  	 = 0.0600;		acc 	= 0.9772
>			time: 140s;		val_loss = 0.0452;		val_acc = 0.9844
'''

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import os
import numpy as np

def main():
	modes = ['training', 'retraining', 'testing', 'none']
	mode = modes[0]
	epoch_num = 3
	epoch_done = 0

	filepath = "dogs_vs_cats"
	# Modify the savepath before each training run.
	savepath = "weightsFile2\\weights3-{epoch:02d}-{loss:.4f}-{val_acc:.2f}.hdf5"
	# Modify the loadpath before each testing run. The number after weights in the filename denotes the run number.
	loadpath = 'weightsFile2\\weights2-03-0.0580-1.00.hdf5'
	num_classes = 2			# dog or cat
	image_size = 224		# dimensions according to resnet model

	#Create checkpoint details
	checkpoint = ModelCheckpoint(savepath, monitor='val_loss', save_best_only=True)
	callbacks_list = [checkpoint]

	#Create model
	model = Sequential()
	model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet')) 	# entire resnet model is the first layer!
	model.add(Dense(num_classes, activation='sigmoid'))
	model.layers[0].trainable = False		# entire resnet layer is not trainable; only the final dense layer is trained
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()

	#Create data generators
	img_generator = image.ImageDataGenerator(preprocessing_function=preprocess_input)
	train_gen = img_generator.flow_from_directory(filepath+"\\training_set",
												  target_size=(image_size, image_size),
												  batch_size=32,
												  class_mode="categorical")
	test_gen = img_generator.flow_from_directory(filepath+"\\test_set",
												 target_size=(image_size, image_size),
												 batch_size=32,
												 class_mode="categorical")
	if mode == 'training':
		#Fit and train model
		model.fit_generator(train_gen,
							steps_per_epoch=250,
							epochs=epoch_num,
							validation_data=test_gen,
							validation_steps=64,
							callbacks=callbacks_list,
							verbose=1)

	elif mode == 'retraining':
		epoch_num -= epoch_done
		model.load_weights(loadpath)
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		print("Model loaded and compiled!", end="\n\n")

		#change the name of file to save as 
		savepath = "weightsFile2\\weights1-contd_%d-{epoch:02d}-{loss:.4f}-{val_acc:.2f}.hdf5" % epoch_done
		#create model checkpoint
		checkpoint = ModelCheckpoint(savepath, monitor='val_loss', save_best_only=True)
		callbacks_list = [checkpoint]
		 
		# fit and train the model
		model.fit_generator(train_gen,
					steps_per_epoch = 250,
					epochs = epoch_num,
					validation_data = test_gen,
					validation_steps = 64,
					callbacks = callbacks_list,
					verbose = 1)

	elif mode == 'testing':
		print("\nThe class indices are: ", train_gen.class_indices, end="\n\n")				# Displays which class (cat or dog) is represented by a 0 and which by 1
		#Load model
		model.load_weights(loadpath)
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		directory = 'testImages'
		for imgfile in os.listdir(directory):
			if imgfile.endswith('.jpg'):			# ensure all images in your testImages folder are in .jpg format, or change this to accept whatever formats you want
				test_image = image.load_img(os.path.join(directory, imgfile), target_size=(image_size, image_size))
				test_image = image.img_to_array(test_image)
				test_image = np.expand_dims(test_image, axis=0)
				result = model.predict(test_image)
				#print(result)
				#print(result[0][0])
				if result[0][0] < result[0][1]:
					# if the prediction value for 1 is more than 0 its a dog (according to train_gen.class_indices)
					prediction = 'dog'
				elif result[0][0] > result[0][1]:
					# if prediction value for 0 is more than 1 its a cat
					prediction = 'cat'
				else:
					# if both prediction values are equal (extremely rare case)
					prediction = 'cat or a dog maybe...'
				print(imgfile, ": I think this is an image of a {0}".format(prediction), end="\n\n")

if __name__ == "__main__":
	main()