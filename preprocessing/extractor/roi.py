
class Roi:
    
    def video_to_frames_roi(input_loc, output_loc_frame, output_loc_roi):
        """Function to extract frames from input video file
        and save them as separate frames in an output directory.
        Args:
            input_loc: Input video file.
            output_loc: Output directory to save the frames.            
        Returns:
            None
        """
        import time
        import cv2
        import os
        
        time_start = time.time()
        
        # Start capturing the feed
        cap = cv2.VideoCapture(input_loc)
        # Find the number of frames
        #print(cv2.CAP_PROP_FRAME_COUNT)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        #print ("Number of frames: ", video_length)
        count = 0
        #print ("Converting video..\n")
        # Start converting the video
        while cap.isOpened():
            # Extract the frame
            ret, frame = cap.read()
            # Write the results back to output location.    
            output_frame = output_loc_frame + "/%#05d.jpg" % (count+1)
            output_roi = output_loc_roi + "/%#05d.jpg" % (count+1)
            print(output_frame)
            print(output_roi)
            cv2.imwrite(output_frame, frame)
            Roi.extract_mouth(output_frame, output_roi)
            count = count + 1
            # If there are no more frames left
            if (count > (video_length)):
                # Log the time again
                time_end = time.time()
                # Release the feed
                cap.release()        
                break


    def lipnet_mouth_extraction(np_mouth_points, frame):
        import numpy as np
        from scipy.misc import imresize

        MOUTH_WIDTH = 100
        MOUTH_HEIGHT = 50
        HORIZONTAL_PAD = 0.19

        normalize_ratio = None

        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
        resized_img = imresize(frame, new_img_shape)

        mouth_centroid_norm = mouth_centroid * normalize_ratio

        mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

        mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
        return mouth_crop_image


    def extract_mouth(image, output_path_image):
        OFFSET_MOUTH_Y = 15
        OFFSET_MOUTH_X = 15

        # import the necessary packages
        from imutils import face_utils
        import numpy as np
        import argparse
        import imutils
        import dlib
        import cv2

        

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        shape_predictor = "extractor/shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(shape_predictor)

        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(image)
        
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)

        with_lipnext = True
        if with_lipnext:
            mouth = Roi.get_frames_mouth(detector, predictor, image)
            cv2.imwrite(output_path_image, mouth)
            return 
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
            # clone = image.copy()
            # cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            #	0.7, (0, 0, 255), 2)

            # loop over the subset of facial landmarks, drawing the
            # specific face part
            # for (x, y) in shape[i:j]:
            #	cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

            # extract the ROI of the face region as a separate image
            
            np_mouth_points = np.array([shape[i:j]])
            print(np_mouth_points)
            (x, y, w, h) = cv2.boundingRect(np_mouth_points)
            height, width, channels = image.shape

            start_y = y - OFFSET_MOUTH_Y
            end_y = y + h + OFFSET_MOUTH_Y

            start_x = x - OFFSET_MOUTH_X
            end_x = x + w + OFFSET_MOUTH_X

            # check doesnt pass the
            if start_y < 0:
                start_y = 0
            if start_x < 0:
                start_x = 0
            if end_y >= height:
                end_y = height - 1
            if end_x >= width:
                end_x = width - 1

            # roi = image[y:y + h, x:x + w]
            roi = Roi.lipnet_mouth_extraction(np_mouth_points[0], image)
            # roi = image[start_y:end_y, start_x:end_x]

            # the resize is important but i need to do it different
            # roi = cv2.resize(roi, (100, 50))
            # roi = imutils.resize(roi, width=100)

            # save mouth
            print(output_path_image)
            # cv2.imshow("ROI", roi)
            # cv2.waitKey(0)
            cv2.imwrite(output_path_image, roi)

    def get_frames_mouth(detector, predictor, frame):
        import numpy as np
        from scipy.misc import imresize

        MOUTH_WIDTH = 100
        MOUTH_HEIGHT = 50
        HORIZONTAL_PAD = 0.19
        normalize_ratio = None
        
        dets = detector(frame, 1)
        shape = None
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            i = -1
        if shape is None: # Detector doesn't detect face, just return as is
            return frames
        mouth_points = []
        for part in shape.parts():
            i += 1
            if i < 48: # Only take mouth region
                continue
            mouth_points.append((part.x,part.y))
        np_mouth_points = np.array(mouth_points)

        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
        resized_img = imresize(frame, new_img_shape)

        mouth_centroid_norm = mouth_centroid * normalize_ratio

        mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

        mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

        return mouth_crop_image
