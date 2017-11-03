###########################################################
# Deep Learning for Advanced Robot Perception
# Assignment # 7-8 Face Recognition
# Convolutional Neural Network for Face Recognition
# prepared by: Katie Gandomi and Kritika Iyer
###########################################################

##################### IMPORT DEPENDENCIES #################
import numpy as np
from numpy import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from PIL import Image
import glob

####################### METHODS #########################
def resize_data(input_path, input_path_resized, img_rows, img_cols):
	# number of channels
	img_channels = 1

	# Get all .jpg input frames
	frames = glob.glob(input_path + "/" + '*.jpg')

	# print frames
	num_samples = size(frames)
	print "Number of samples to resize: ", num_samples

	# For every frame in the input set convert the data to grayscale and resize
	for frame in frames:
		im = Image.open(frame)  
		img = im.resize((img_rows,img_cols))
		gray = img.convert('L')
		gray.save(input_path_resized + "/" + frame[len(input_path)+1:], "JPEG")

	print "Resize Complete!"

def load_data(input_path_resized, img_rows, img_cols):
	# Get the number of resized, grayscale, images in new path
	input_frames = glob.glob(input_path_resized + "/" + '*.jpg')
	input_num_samples = len(input_frames)
	print "Number of input samples: ", input_num_samples

	# create matrix to store all flattened images
	immatrix = array([array(Image.open(input_path_resized + "/" + input_frame[len(input_path_resized)+1:])).flatten() for input_frame in input_frames],'f')

	# hardcode the labels for the data set
	label=np.ones((input_num_samples,),dtype = int)
	label[0:3648]=0 # kritika
	label[3648:]=1 # katie
	number_of_labels = 2 # for kritika and katie <---------- HARDCODED !

	# Now Shuffle all the data and return the values
	data,Label = shuffle(immatrix,label, random_state=2)
	main_data = [data,Label]

	# Seperate main data out
	(X, y) = (main_data[0], main_data[1])

	# Create the test and training data by splitting them
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

	# Reshape the x_data
	X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	X_train /= 255
	X_test /= 255

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, number_of_labels)
	Y_test = np_utils.to_categorical(y_test, number_of_labels)

	# Return the data
	return X_train, Y_train, X_test, Y_test

def create_model(img_rows, img_cols):
	number_of_labels = 2 # for kritika and katie <---------- HARDCODED !
	# Create CNN model
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, dim_ordering="th", input_shape=(1, img_rows, img_cols), activation='relu'))
	model.add(Dropout(0.2))
	model.add(Convolution2D(32, 3, 3, dim_ordering="th", activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(number_of_labels, activation='softmax'))

	return model

def compile_model(model, X_train, Y_train, X_test, Y_test):
	# Set up parameters for compiling the model
	epochs = 25
	lrate = 0.01
	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

	# Compile the model
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	print(model.summary())

	# Fit the model
	history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=epochs, batch_size=32)

	# Final evaluation of the model
	scores = model.evaluate(X_test, Y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	return history, scores

def plot_model(history, scores):
	# Plot Results
	print "Plotting..."
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

#################################################################
#################### MAIN : CODE RUNS HERE ######################
#################################################################
# path to folder of images
input_path = "data"
input_path_resized = "data_resized"
img_rows, img_cols = 169, 300 # Size of resized images

# resize_data(input_path, input_path_resized, img_rows, img_cols)

X_train, Y_train, X_test, Y_test = load_data(input_path_resized, img_rows, img_cols)
print "Data Loaded Sucessfully!"

# model = create_model(img_rows, img_cols)
print "Model Created"

# history, scores = compile_model(model, X_train, Y_train, X_test, Y_test)
print "Model Compiled"

# plot_model(history, scores)







