import cv2
import os
import numpy as np
maximages = 9000
for i in [4]:
	num = len(os.listdir(str(i)))
	newnum = num
	while newnum < maximages:
		img = cv2.imread(str(i)+'/'+str(np.random.randint(num))+'.jpg')
		img = cv2.flip(img,1)
		cv2.imwrite(str(i)+'/'+str(newnum)+'.jpg',img)
		newnum = newnum + 1
		# break
	# break
#8604 9000 7610 8983