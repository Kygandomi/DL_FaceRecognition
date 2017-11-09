###########################################################
# Deep Learning for Advanced Robot Perception
# Assignment # 7-8 Face Recognition
# Parsing Video data into Frames
# prepared by: Katie Gandomi and Kritika Iyer
###########################################################

# Import Dependencies
import os
import cv2
import glob
import imutils

# Set up directort to save frames to and to get videos from
name = "spjain" #change name for different person
vidpath = name + '_videos' #videos savend in same folder as code with name_videos folder
newpath = name # frame folder with name of person

# Setup path to save data in the current directory
if not os.path.exists(newpath):
    os.makedirs(newpath)

# Get all videos in vidpath
videos = glob.glob(vidpath + "/" + '*.mp4')
# print (videos)

# Loop over each video and save the frames
total_frames = 0
for video in videos:
    print ("Processing Video...")
	# count = 0 # Used for labeling the frames

	# Make a path for the frames to be written too
    frame_path = newpath + "/" + video[len(vidpath)+1:-4]
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
	# Open video and read first frame
    video_capture = cv2.VideoCapture(video)
    success,image = video_capture.read()
    # print(success)

	# While we're reading frames, save them to the folder
    while success:
        rotated = imutils.rotate_bound(image, 90)
        cv2.imwrite(newpath + "/" + name + "frame%d.jpg" % total_frames, rotated)     # save frame as JPEG file
        success,image = video_capture.read()
        total_frames += 1



# Process Completed
print ("Process Completed for ", name)
print ("Total Frames: ", total_frames)
