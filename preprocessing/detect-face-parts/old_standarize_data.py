from sklearn import preprocessing
from sklearn import preprocessing
import os
import cv2
import glob
import argparse
from scipy import misc

from sklearn.datasets import load_iris

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
from time import time
from time import sleep

folder = "pruebas/roi_frames/videos/s1"

onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

print("Working with {0} images".format(len(onlyfiles)))
print("Image examples: ")

for i in range(40, 42):
    print(onlyfiles[i])
    display(_Imgdis(filename=folder + "/" + onlyfiles[i], width=240, height=320))

# Create folder for rois
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_all_images(path_rois):
    X_data = []
    for subdir, dirs, files in os.walk(path_rois): 
        for file in files:   
            path_single_roi = os.path.join(subdir, file)            
            image = cv2.imread (path_single_roi)            
            #print(image)
            X_data.append (image)
    return X_data

def save_standarize_images(path_standarize_rois, X_scaled):
    count = 0
    for x in X_scaled:
        output_frame = path_standarize_rois + "/%#05d.jpg" % (count+1)
        print(output_frame)
        cv2.imwrite(output_frame, x)
        count = count + 1


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--rois", required=True,
	help="path to rois")
args = vars(ap.parse_args())

path_rois = args["rois"]

path_standarize_rois = "pruebas/standarize_rois"

make_dir(path_rois)

X_train = read_all_images(path_rois)

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                     [ 0.,  1., -1.]])

# load the Iris dataset
iris = load_iris()
##print(iris.data.shape)
# separate the data and target attributes
X = iris.data

print(X)

X_scaled = preprocessing.scale(X)

print(X_scaled)

#save_standarize_images(path_standarize_rois, X_scaled)
