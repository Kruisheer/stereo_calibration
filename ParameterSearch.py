from matplotlib import pyplot as plt
#%matplotlib inline
import imutils
import cv2
import numpy as np


cap_right = cv2.VideoCapture('cam1_undistorted.mp4')
cap_left  = cv2.VideoCapture('cam2_undistorted.mp4')

stereo = cv2.StereoBM_create(numDisparities= 16, blockSize= 15)

#Sync the frames as close as possible
for i in range(1):
	ret, right_image = cap_right.read()


for j in range(16):
	ret, left_image = cap_left.read()

while(True):

	ret, right_image = cap_right.read()
	ret, left_image = cap_left.read()
	
	right_Gry = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
	left_Gry  = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
	disparity = stereo.compute(left_Gry , right_Gry)
	#disparity = stereo.compute(right_Gry, left_Gry)

	# right_image = imutils.resize(right_image, width=int(1920/2))
	# cv2.imshow('right_cam1', right_image)
	# left_image = imutils.resize(left_image, width=int(1920/2))
	# cv2.imshow('left_cam2', left_image)

	disparity = imutils.resize(disparity, width=int(1920/2))
	cv2.imshow('disparity', disparity)


	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()

