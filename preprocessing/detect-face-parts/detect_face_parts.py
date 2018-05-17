# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 


def extractMouth(image, outputfolder):
	# import the necessary packages
	from imutils import face_utils
	import numpy as np
	import argparse
	import imutils
	import dlib
	import cv2

	shape_predictor = "shape_predictor_68_face_landmarks.dat"

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor)

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(image)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		onlyOnce = True
		# obtain the mouth the mouth is in the index 0 look https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/helpers.py
		(name, (i, j)) = list(face_utils.FACIAL_LANDMARKS_IDXS.items())[0]
		# clone the original image so we can draw on it, then
		# display the name of the face part on the image
		clone = image.copy()
		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)

		# loop over the subset of facial landmarks, drawing the
		# specific face part
		for (x, y) in shape[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)



		# extract the ROI of the face region as a separate image
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		roi = image[y:y + h, x:x + w]
		roi = cv2.resize(roi, (100, 50))
		#roi = imutils.resize(roi, width=100)

		# save mouth
		cv2.imwrite("roi/0.jpg", roi)

extractMouth("images/0.jpg","roi")

		