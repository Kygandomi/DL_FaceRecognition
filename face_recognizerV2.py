#!/bin/env python3
# SBATCH -N 1 # No. of computers you wanna use. Typically 1
# SBATCH -n 2 # No. of CPU cores you wanna use. Typically 1
# SBATCH -p gpu # This flag specifies that you wanna use GPU and not CPU
# SBATCH -o homework.out # output file name, in case your program has anything to output (like print, etc)
# SBATCH -t 24:00:00 # Amount of time
# SBATCH --gres=gpu:3 # No. of GPU cores you wanna use. Usually 2-3

###########################################################
# Deep Learning for Advanced Robot Perception
# Assignment # 7-8 Face Recognition
# Convolutional Neural Network for Face Recognition
# prepared by: Katie Gandomi, Kritika Iyer, Shubham jain
###########################################################

##################### IMPORT DEPENDENCIES #################
import numpy as np
from numpy import *
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import LearningRateScheduler
from keras.layers import Activation
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from PIL import Image
import pickle
import glob

####################### METHODS #########################

def resize_data(input_path, input_path_resized, img_rows, img_cols):
	# number of channels
	img_channels = 1

	# Get all .jpg input frames
	frames = glob.glob(input_path + "/" + '*.jpg')

	# print frames
	num_samples = size(frames)
	print ("Number of samples to resize: %d", num_samples)

	# For every frame in the input set convert the data to grayscale and resize
	for frame in frames:
		im = Image.open(frame)
		img = im.resize((img_rows,img_cols))
		gray = img.convert('L')
		gray.save(input_path_resized + "/" + frame[len(input_path)+1:], "JPEG")

	print ("Resize Complete!")

def load_data():
	# Get the number of resized, grayscale, images in new path
	# input_frames = glob.glob(input_path_resized + "/" + '*.jpg')
	# input_num_samples = len(input_frames)
	# print ("Number of input samples:", input_num_samples)

	# create matrix to store all flattened images
	# immatrix = array([array(Image.open(input_path_resized + "/" + input_frame[len(input_path_resized)+1:])).flatten() for input_frame in input_frames],'f')

	immatrix,label= pickle.load(open('database_mid.pickle','rb'))
	print ("Number of input samples:", len(label))
	print ("images:",immatrix.shape)
	label=np.asarray(label)
	label[0:9000]=0 #spjain
	label[9000:18000]=1 #ajram
	label[18000:27000]=2 #pbhaskaran
	label[27000:36000]=3 #kygandomi
	label[36000:45000]=4 #kiyer
	label=np.asarray(label,dtype=np.int16)
	print (label[0])
	print (label[9000])
	print (label[18000])
	print (label[27000])
	print (label[36000])

	# hardcode the labels for the data set
	# label=np.ones((input_num_samples,),dtype = int)
	# label[0:3648]=0 # kritika
	# label[3648:]=1 # katie
	# # create more lables here
	immatrix=np.reshape(immatrix,(45000,1,170,96))
	number_of_labels = 5 # for kritika and katie <---------- HARDCODED !

	# Now Shuffle all the data and return the values
	data,Label = shuffle(immatrix,label, random_state=2)
	main_data = [data,Label]
	# print (Label[0])
	# Seperate main data out
	(X, Y) = (main_data[0], main_data[1])

	# Create the test and training data by splitting them
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=4)

	# Reshape the x_data
	# X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	# X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	#
	#X_train = X_train.astype('float32')
	#X_test = X_test.astype('float32')
	#
	#X_train /= 255
	#X_test /= 255

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(Y_train, number_of_labels)
	Y_test = np_utils.to_categorical(Y_test, number_of_labels)
	print (X_train.shape)
	print (Y_train.shape)
	print (X_test.shape)
	print (Y_test.shape)
	print (Y_train[1:10])
	# Return the data
	return X_train, Y_train, X_test, Y_test

def create_model():
	number_of_labels = 5 # for kritika and katie, shubham arjun and prakash <---------- HARDCODED !
	# Create CNN model
	model = Sequential()
	model.add(Convolution2D(16, 3, 3, dim_ordering="th", input_shape=(1, 170, 96), activation='relu'))
	model.add(Convolution2D(16, 3, 3, dim_ordering="th"))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(32, 3, 3, dim_ordering="th", activation='relu'))
	model.add(Convolution2D(32, 3, 3, dim_ordering="th"))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(32, 3, 3, dim_ordering="th", activation='relu'))
	model.add(Convolution2D(32, 3, 3, dim_ordering="th"))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(16, 3, 3, dim_ordering="th", activation='relu'))
	model.add(Convolution2D(16, 3, 3, dim_ordering="th"))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(number_of_labels, activation='softmax'))
	return model

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 5.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def compile_model(model, X_train, Y_train, X_test, Y_test):
	# Set up parameters for compiling the model
	epochs = 16
	#lrate = 0.001
	#decay = lrate/epochs
	sgd = SGD(lr=0, momentum=0.9, decay=0, nesterov=False)
	adam = Adam(lr=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	# Compile the model
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	print(model.summary())
	lrate = LearningRateScheduler(step_decay)
	callbacks_list = [lrate]
	# Fit the model
	history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=epochs, callbacks = callbacks_list, batch_size=64)

	# Final evaluation of the model
	scores = model.evaluate(X_test, Y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	return history, scores,model

# def plot_model(history, scores):
	# # Plot Results
	# print ("Plotting...")
	# f=open('data.txt,'w')
	# f.write("accuracy,ValAcc, Loss, ValLoss \n")
	# for i in range(len(history.history['acc'])):
	# 	f.write(str(history.history['acc'][i]) + "," + str(history.history['val_acc'][i]) + "," + str(history.history['loss'][i]) + "," + str(history.history['val_loss'][i]) + "\n")
	# f.close()
	# print "Done!"
#################################################################
#################### MAIN : CODE RUNS HERE ######################
#################################################################
# path to folder of images
# input_path = "data"
# input_path_resized = "data_resized"
# img_rows, img_cols = 169, 300 # Size of resized images

# resize_data(input_path, input_path_resized, img_rows, img_cols)

X_train, Y_train, X_test, Y_test = load_data()
print ("Data Loaded Sucessfully!")

model = create_model()
print ("Model Created")

history, scores,model = compile_model(model, X_train, Y_train, X_test, Y_test)
model.save_weights("model.h5")
print ("Model Compiled")
f = open('output_history.pickle','wb')
pickle.dump(history.history,f)
f.close()
f = open('output_scores.pickle','wb')
pickle.dump(scores,f)
f.close()
#plot(history)
#plot_model(history, scores)
