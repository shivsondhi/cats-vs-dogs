'''
Cats and Dogs - Image Classification Task using Python, Keras and TensorFlow.

TRIAL 1 :	125 steps / epoch, with batch size = 32
			epoch = 1; 		loss 	 = 0.7071;		acc 	= 0.5707
			time: 231s; 	val_loss = 0.6337; 		val_acc = 0.6512

			epoch = 5; 		loss 	 = 0.5774; 		acc 	= 0.7010
			time: 105s; 	val_loss = 0.6002; 		val_acc = 0.6836

			epoch = 10; 	loss 	 = 0.5355; 		acc 	= 0.5259
			time: 164s; 	val_loss = 0.5259; 		val_acc = 0.7364

			epoch = 15; 	loss 	 = 0.5025; 		acc 	= 0.7575
			time: 122s; 	val_loss = 0.5756; 		val_acc = 0.7208

			epoch = 20; 	loss 	 = 0.4890; 		acc 	= 0.7618
			time: 153s; 	val_loss = 0.5119; 		val_acc = 0.7549
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import os

def main():
	modes = ['training', 'retraining', 'testing', 'none']
	mode = modes[1]
	epoch_num = 20		# total epochs to be done
	epoch_done = 0		# number of epochs done already if retraining

	print("We are in the {0} phase".format(mode), end="\n\n")

	#CREATE THE MODEL
	#initialize the CNN 
	classifier = Sequential()
	#Step 1 - Convolution
	classifier.add(Conv2D(32, (3, 3), input_shape=(3, 64, 64), activation='relu'))
	#Step 2 - Pooling
	classifier.add(MaxPooling2D(pool_size=(2,2)))
	#Step 3 - Flattening
	classifier.add(Flatten())
	#Step 4 - Full Connection
	classifier.add(Dense(128, activation='relu'))
	classifier.add(Dense(1, activation='sigmoid'))
	print('Model built!')
	#Step 5 - Compiling the CNN
	classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	print("Model compiled!", end="\n\n")
	print(classifier.summary())

	#create data generator object
	train_datagen = image.ImageDataGenerator(
						rescale = 1./255,
						shear_range = 0.2,
						zoom_range = 0.2,
						horizontal_flip = True)
	print("train_data image augmentation done.")
	test_datagen = image.ImageDataGenerator(rescale=1./255)
	print("test_data image augmentation done.", end="\n\n")

	training_set = train_datagen.flow_from_directory(
						'dogs_vs_cats\\training_set',
						target_size = (64, 64),
						batch_size = 32,
						class_mode = 'binary')
	print("training data batch generator created.")
	test_set = test_datagen.flow_from_directory(
						'dogs_vs_cats\\test_set',
						target_size = (64, 64),
						class_mode = 'binary')
	print("test data batch generator created.", end="\n\n")

	if mode == "training":
		#name of file to save as
		filepath = "weightsFile1\\first_weights-{epoch:02d}-{loss:.4f}-{val_acc:.2f}.hdf5"
		#create a checkpoint to callback at after every epoch
		checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True)
		callbacks_list = [checkpoint]
		#fit the model
		print("Starting training...")
		classifier.fit_generator(
						training_set,
						steps_per_epoch = 125,
						epochs = epoch_num,
						validation_data = test_set,
						validation_steps = 1000,
						verbose = 1,
						callbacks = callbacks_list)
		print("\n\t\t~Fin~\n")
	
	elif mode == "retraining":
		epoch_num -= epoch_done
		# load the previous best weights manually
		best_file = "weightsFile1\\first_weights-contd_10-07-0.4837-0.73.hdf5"
		classifier.load_weights(best_file)
		classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		print("Model loaded and compiled!", end="\n\n")
		# name of file to save as 
		filepath = ("weightsFile1\\first_weights-contd_%d-{epoch:02d}-{loss:.4f}-{val_acc:.2f}.hdf5", epoch_done)
		#create model checkpoint
		checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True)
		callbacks_list = [checkpoint] 
		# fit the model
		print("Starting training...")
		classifier.fit_generator(training_set,
					steps_per_epoch = 125,
					epochs = epoch_num,
					validation_data = test_set,
					validation_steps = 1000,
					verbose = 1,
					callbacks = callbacks_list
					)
		print("\n\t\t~Fin~\n")
	
	elif mode == "testing":
		#load model weights
		best_file = "weightsFile1\\first_weights-contd_10-07-0.4837-0.73.hdf5"
		classifier.load_weights(best_file)
		classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		print("Model loaded and compiled!", end="\n\n")
		#iterate through the testImages folder and make predictions on each .jpg file
		directory = 'testImages'
		for filename in os.listdir(directory):
			if filename.endswith(".jpg"):
				#load and preprocess the image to classify
				test_image = image.load_img(os.path.join(directory, filename), target_size=(64, 64))
				test_image = image.img_to_array(test_image)
				test_image = np.expand_dims(test_image, axis=0)
				result = classifier.predict(test_image)
				training_set.class_indices
				if result[0][0] >= 0.5:
					prediction = 'dog'
				else:
					prediction = 'cat'
				print(filename, ": I think this is an image of a {0}!".format(prediction), end="\n\n")
		print("\n\t\t~Fin~\n")
	else:
		print("\n\t\t~Fin~\n")

if __name__ == "__main__":
	main()