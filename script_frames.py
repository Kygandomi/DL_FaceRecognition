import cv2
import os
read_folder = 'kygandomi_frames/'
out_folder = '1/'
count = 0
frames = os.listdir(read_folder)
for frame in frames:
	# print(frame)
	count = count+1
	num = int(frame.split('.')[0][14:])
	img = cv2.imread(read_folder+frame)
	# if num >= 3649:
	# 	img = cv2.flip(img,0)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img = cv2.resize(img, (192, 341))
	cv2.imwrite(out_folder+str(num)+'.jpg',img)
	if count%500==0:
		print(count)
	# break
