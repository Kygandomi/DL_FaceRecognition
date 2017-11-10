import cv2
import os
import numpy as np
read_folder = 'Arjun/'
out_folder = '4/'
count = 0
videos = os.listdir(read_folder)
for video in videos:
	vid = cv2.VideoCapture(read_folder+video)
	success,img = vid.read()
	while success:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = np.rot90(img)
		img = cv2.resize(img, (192, 341))
		cv2.imwrite(out_folder+str(count)+'.jpg',img)
		if count%500==0:
			print(count)
		count = count+1
		success,img = vid.read()
		# break

