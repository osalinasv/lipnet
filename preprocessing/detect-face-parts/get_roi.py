from video2frames import video_to_frames_roi

import argparse

import os

# try it with get_roi.py -v pruebas/videos

# Create folder for rois
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--videos", required=True,
	help="path to videos")
args = vars(ap.parse_args())

path_videos = args["videos"]

path_frames = "pruebas/frames"

path_rois = "pruebas/roi_frames"

make_dir(path_frames)

make_dir(path_rois)

for subdir, dirs, files in os.walk(path_videos):    
    start = subdir.find('/') + 1        
    # s1, s2, .. ,sn
    sub = subdir[start:]

    for file in files:               
        # get the name of the image without the extension
        base = os.path.splitext(file)[0]    

        path_single_video = os.path.join(subdir, file)
        path_image_frames = os.path.join(path_frames, sub, base) 
        path_roi_frames = os.path.join(path_rois, sub, base) 
        
        print(path_single_video)
        print(path_image_frames)
        print(path_roi_frames)

        make_dir(path_image_frames)
        make_dir(path_roi_frames)

        video_to_frames_roi(path_single_video,path_image_frames,path_roi_frames)     

