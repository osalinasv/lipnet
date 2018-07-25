import dlib
import os
import sys

from preprocessing.extractor.extract_roi import extract_video_data


predictor_path = os.path.realpath(os.path.join(__file__, '..', '..', 'data', 'predictors', 'shape_predictor_68_face_landmarks.dat'))

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

video_path = os.path.realpath(sys.argv[1])
video_data = extract_video_data(video_path, detector, predictor)

if video_data is not None:
	print(video_data.shape)
