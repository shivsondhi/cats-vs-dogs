'''
Cats and Dogs - Image Classification using ResNet50 Transfer Learning.
4000 training images of cats and dogs each + 1000 test images each.

TRIAL 1 : 	125 steps/epoch + batch_size of 32	||	categorical_crossentropy + softmax
>			epoch = 1; 		loss	 = 0.1875;		acc 	= 0.9223
>			time: 117s;		val_loss = 0.0188; 		val_acc = 1.0000
>
>			epoch = 2;		loss 	 = 0.0821;		acc 	= 0.9677
>			time: 124s; 	val_loss = 0.0252;		val_acc = 1.0000
>
>			epoch = 3; 		loss 	 = 0.0737; 		acc 	= 0.9725
>			time: 114s;		val_loss = 0.0145; 		val_acc = 1.0000

TRIAL 2 : 	125 steps/epoch + batch_size of 32 	|| 	binary_crossentropy + sigmoid
>			epoch = 1;		loss 	 = 0.1475;		acc 	= 0.9373
>			time: 143s; 	val_loss = 0.0173;		val_acc = 1.0000
>
>			epoch = 1;		loss 	 = 0.0750;		acc 	= 0.9698
> 			time: 192s;		val_loss = 0.0036; 		val_acc = 1.0000
>
>			eopch = 3;		loss  	 = 0.0580;		acc 	= 0.9760
>			time: 213s;		val_loss = 0.0057;		val_acc = 1.0000
> 			total-time: 595.3s.
'''

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import os
import numpy as np

def main():
	modes = ['training', 'testing']
	mode = modes[0]

	filepath = "dogs_vs_cats"
	# Modify the savepath before each training run.
	savepath = "weightsFile2\\weights3-{epoch:02d}-{loss:.4f}-{val_acc:.2f}.hdf5"
	# Modify the loadpath before each testing run. The number after weights in the filename denotes the run number.
	loadpath = 'weightsFile2\\weights2-03-0.0580-1.00.hdf5'
	num_classes = 2			# dog or cat
	image_size = 224		# dimensions according to resnet model

	#Create checkpoint details
	checkpoint = ModelCheckpoint(savepath, monitor='loss', save_best_only=True)
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
												 class_mode="categorical")
	if mode == 'training':
		#Fit and train model
		model.fit_generator(train_gen,
							steps_per_epoch=125,
							epochs=3,
							validation_data=test_gen,
							validation_steps=1,
							callbacks=callbacks_list,
							verbose=1)
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