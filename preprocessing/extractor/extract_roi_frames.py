import os
import cv2
import dlib
import numpy as np
import skvideo.io

from imutils import face_utils
from progress.bar import ShadyBar
from scipy.misc import imresize
from skimage import io

MOUTH_WIDTH = 100
MOUTH_HEIGHT = 50
HORIZONTAL_PAD = 0.10


def video_to_frames(file_path: str, output_dir: str, predictor_path: str):
    """"
    :return
        false in case no mouth is detected in a frame
        true if everything is rigth
    """
    video_object = skvideo.io.vread(file_path)
    frames = np.array([frame for frame in video_object])

    print('\nProcessing: {}'.format(file_path))
    suffix = '%(percent)d%% [%(elapsed_td)s]'
    bar = ShadyBar(os.path.basename(file_path), max=len(frames), suffix=suffix)

    for i, frame in enumerate(frames):
        filename = '{:05}.jpg'.format(i + 1)
        output_cutout_file_path = os.path.join(output_dir, filename)

        mouth = extract_mouth(frame, predictor_path)
        if mouth is None:
            return False
        io.imsave(output_cutout_file_path, mouth)
        bar.next()
    bar.finish()
    return True


def extract_mouth(frame, predictor_path: str):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    for i, rect in enumerate(detector(gray, 1)):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Obtain the mouth landmark at index 0
        # See: https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/helpers.py
        (_, (i, j)) = list(face_utils.FACIAL_LANDMARKS_IDXS.items())[0]

        # Extract the ROI of the face region as a separate image
        np_mouth_points = np.array([shape[i:j]])

        return crop_mouth_region(np_mouth_points[0], frame)


def crop_mouth_region(np_mouth_points, frame):
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

    # This is wrong is still making the
    # TODO: i make the chance but i have to be one hundred porcent sure
    diff_width = mouth_r - mouth_l
    if diff_width > MOUTH_WIDTH:
        mouth_r += MOUTH_WIDTH - diff_width

    diff_height = mouth_b - mouth_t
    if diff_height > MOUTH_HEIGHT:
        mouth_b += MOUTH_HEIGHT - diff_height

    mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
    return mouth_crop_image
