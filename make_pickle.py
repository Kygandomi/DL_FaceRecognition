import pickle
import os
import cv2
import numpy as np
data_folder = 'training_data/'
labels = os.listdir(data_folder)
imsize = (85,48,0)
X = np.array([], dtype=np.uint8).reshape(170,96,0)
y = []
temp = np.array([], dtype=np.uint8).reshape(170,96,0)
for label in labels:
	images = os.listdir(data_folder+label)
	for i in range(9000):
		img = cv2.imread(data_folder+label+'/'+str(i)+'.jpg')
		img = cv2.resize(img, (96, 170))
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		temp = np.dstack((temp,img))
		y.append(label)
		if(i%500 == 0):
			print(i)
			X = np.dstack((X,temp))
			temp = np.array([], dtype=np.uint8).reshape(170,96,0)
	X = np.dstack((X,temp))
	temp = np.array([], dtype=np.uint8).reshape(170,96,0)
	

print(X.shape)
print(len(y))
print(y[0])
f = open('database_mid.pickle','wb')
pickle.dump([X,y],f)
f.close()