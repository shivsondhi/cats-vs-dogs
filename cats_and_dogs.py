'''
Cats and Dogs - Image Classification Task using Python, Keras and TensorFlow.

TRIAL 1 :	7000 steps / epoch
			epoch = 1;		loss 	 = 0.4053;		acc 	= 0.8081
			time: 1885s;	val_loss = 0.1738;		val_acc = 0.9318

			epoch = 2;		loss 	 = 0.163;		acc 	= 0.9358
			time: 1602s;	val_loss = 0.067;		val_acc = 0.9780

			epoch = 3;		loss 	 = 0.0846;		acc 	= 0.9693
			time: 1128s;	val_loss = 0.0396;		val_acc = 0.9856

			epoch = 4; 		loss 	 = 0.0593;		acc 	= 0.9795
			time: 1126s;	val_loss = 0.0305;		val_acc = 0.9891

			epoch = 5; 		loss 	 = 0.0456; 		acc 	= 0.9844
			time: 1098s; 	val_loss = 0.0399; 		val_acc = 0.9861
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import os

def main():
	modes = ['training', 'testing', 'none']
	mode = modes[1]
	epoch_num = 5

	print("We are in the {0} phase".format(mode), end="\n\n")

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
						'dogs_vs_cats\\training_set',
						target_size = (64, 64),
						batch_size = 32,
						class_mode = 'binary')
	print("test data batch generator created.", end="\n\n")

	if mode == "training":
		#name of file to save as
		filepath = "weightsFile1\\first_weights-{epoch:02d}-{loss:.4f}-{val_acc:.2f}.hdf5"
		#create a checkpoint to callback at after every epoch
		checkpoint = ModelCheckpoint(filepath, period=1)
		callbacks_list = [checkpoint]
		#fit the model
		print("Starting training...")
		classifier.fit_generator(
						training_set,
						steps_per_epoch = 7000,
						epochs = epoch_num,
						validation_data = test_set,
						validation_steps = 800,
						verbose = 1,
						callbacks = callbacks_list)
		print("\n\t\t~Fin~\n")
	elif mode == "testing":
		#load model weights
		best_file = "weightsFile1\\first_weights-05-0.0456-0.99.hdf5"
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