import os

import numpy as np  # linear algebra
from IPython.display import Image as _Imgdis, display
from sklearn import preprocessing

folder = "pruebas/roi_frames/videos/s1/prwd4n/"

onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

print("Working with {0} images".format(len(onlyfiles)))
print("Image examples: ")

for i in range(40, 42):
    print(onlyfiles[i])
    display(_Imgdis(filename=folder + "/" + onlyfiles[i], width=240, height=320))

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

train_files = []
# y_train = []
i = 0
for _file in onlyfiles:
    train_files.append(_file)
    # label_in_file = _file.find("_")
    # y_train.append(int(_file[0:label_in_file]))

print("Files in train_files: %d" % len(train_files))

# Original Dimensions
image_width = 100
image_height = 50
ratio = 4

# image_width = int(image_width / ratio)
# image_height = int(image_height / ratio)

channels = 3
nb_classes = 1

dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
                     dtype=np.float32)
dataset_flat = np.ndarray(shape=(len(train_files), channels * image_height * image_width),
                          dtype=np.float64)

i = 0
for _file in train_files:
    img = load_img(folder + "/" + _file)  # this is a PIL image
    img.thumbnail((image_width, image_height))
    # Convert to Numpy Array
    x = img_to_array(img)
    # x = x.reshape((3, image_height, image_width))
    x_plane = x.reshape((3 * image_height * image_width))
    # Normalize
    # x = (x - 128.0) / 128.0
    dataset[i] = x
    dataset_flat[i] = x_plane
    i += 1
    if i % 250 == 0:
        print("%d images to array" % i)
print("All images to array!")

print(dataset_flat[0])

print("Scaled")
X_scaled = preprocessing.scale(dataset_flat)

datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

print(dataset[0])

datagen.fit(dataset)

print("Dataset standarized")

for X_batch1, Y_batch1 in datagen.flow(dataset, batch_size=20):
    print(X_batch1[0])

# X_scaled = (dataset - dataset.mean()) / dataset.std()
# print(X_scaled)


X_p = np.array([[1., -1., 2.],
                [2., 0., 0.],
                [0., 1., -1.]])

X_s_library = preprocessing.scale(X_p)

X_s_mine = (X_p - X_p.mean()) / X_p.std()

# print(normalizes.mean(axis=0))

# print(X_p)

# print(X_scaled.mean(axis=0))
# print(X_scaled.std(axis=0))
