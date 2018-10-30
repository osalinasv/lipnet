# lipnet

> A Keras implementation of LipNet

This is an implementation of the spatiotemporal convolutional neural network described by Assael et al. in [this article](https://arxiv.org/abs/1611.01599). However, this implementation only tests the unseen speakers task, the overlapped speakers task is yet to be implemented.

The best training completed yet was started the 26th of September, 2018:

|        Task       | Epochs |  CER   |  WER   |
|:-----------------:|:------:|:------:|:------:|
|  Unseen speakers  |   70   |  9.3%  | 15.7%  |

## Setup

### Prerequisites

Go to [Python's official site](http://python.org) to download and install Python version 3.6.6. If in a Unix/Linux system, follow your package manager's instructions to install the correct version of Python, some distros might already have such version. This project has not been tested in higher Python versions and it might not work properly.

If using with TensorFlow GPU, follow [TensorFlow's](https://www.tensorflow.org/install/gpu) and [NVIDIA's CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) installation guides. This proyect was tested with TensorFlow GPU 1.10.0 and CUDA 9.0.

### Installation

To install all dependencies run the following command:

```
pip install -r requirements.txt

Depending in your Python environment the pip command might be different.
```

If you do not plan to use TensorFlow or TensorFlow GPU, remember to comment out and replace the line `tensorflow-gpu==1.10.0` with your Keras back-end of choice.

## Usage

### Preprocessing

This project was trained using the [GRID corpus dataset](http://spandh.dcs.shef.ac.uk/gridcorpus/) as per the original article.

Given the following directory structure:

```
GRID:
├───s1
│   ├───bbaf2n.mpg
│   ├───bbaf3s.mpg
│   └───...
├───s2
│   └───...
└───...
    └───...
```

Use the `preprocesing/extract.py` script to process all videos into `.npy` binary files if the extracted lips. By default, each file has a numpy array of shape (75, 50, 100, 3). That is 75 frames each with 100 pixels in width and 50 in height with 3 channels per pixel.

```
usage: extract.py [-h] -v VIDEOS_PATH -o OUTPUT_PATH [-pp PREDICTOR_PATH]
                  [-p PATTERN] [-fv FIRST_VIDEO] [-lv LAST_VIDEO]

optional arguments:
  -h, --help            show this help message and exit
  -v VIDEOS_PATH, --videos-path VIDEOS_PATH
                        Path to videos directory
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path for the extracted frames
  -pp PREDICTOR_PATH, --predictor-path PREDICTOR_PATH
                        (Optional) Path to the predictor .dat file
  -p PATTERN, --pattern PATTERN
                        (Optional) File name pattern to match
  -fv FIRST_VIDEO, --first-video FIRST_VIDEO
                        (Optional) First video index extracted in each speaker
                        (inclusive)
  -lv LAST_VIDEO, --last-video LAST_VIDEO
                        (Optional) Last video index extracted in each speaker
                        (exclusive)
```

i.e:

```
python preprocessing\extract.py -v GRID -o data\dataset
```

This results in a new directory with the preprocessed dataset:

```
dataset:
├───s1
├───s2
└───...
```

The original article excluded speakers S1, S2, S20 and S22 from the training dataset.

### Training

Use the `train.py` script to start training a model after preprocesing your dataset. You'll also need to provide a directory containing individual align files with the expected sentence:

```
usage: train.py [-h] -d DATASET_PATH -a ALIGNS_PATH [-e EPOCHS] [-ic]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET_PATH, --dataset-path DATASET_PATH
                        Path to the dataset root directory
  -a ALIGNS_PATH, --aligns-path ALIGNS_PATH
                        Path to the directory containing all align files
  -e EPOCHS, --epochs EPOCHS
                        (Optional) Number of epochs to run
  -ic, --ignore-cache   (Optional) Force the generator to ignore the cache
                        file
```

i.e:

```
python train.py -d data/dataset -a data/aligns -e 70
```

The training is configured to use multiprocessing with 2 workers by default.

Before starting the training, the dataset in the given directory is split by 20% for validation and 80% for training. A cache file of this split is saved inside the `data` directory, i.e: `data/dataset.cache`.

### Evaluating

Use the `predict.py` script to analyze a video or a directory of videos with a trained model:

```
usage: predict.py [-h] -v VIDEO_PATH -w WEIGHTS_PATH [-pp PREDICTOR_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -v VIDEO_PATH, --video-path VIDEO_PATH
                        Path to video file or batch directory to analize
  -w WEIGHTS_PATH, --weights-path WEIGHTS_PATH
                        Path to .hdf5 trained weights file
  -pp PREDICTOR_PATH, --predictor-path PREDICTOR_PATH
                        (Optional) Path to the predictor .dat file
```

i.e:

```
python predict.py -w data/res/2018-09-26-02-30/lipnet_065_1.96.hdf5 -v data/dataset_eval
```

### Configuration

The `env.py` file hosts a number of configurable variables:

Related to the videos:
- **FRAME_COUNT:** The number of frames to be expected for each video
- **IMAGE_WIDTH:** The width in pixels for each video frame
- **IMAGE_HEIGHT:** The height in pixels for each video frame
- **IMAGE_CHANNELS:** The amount of channels for each pixel (3 is RGB and 1 is greyscale)

Related to the neural net:
- **MAX_STRING:** The maximum amount of characters to expect as the encoded align sentence vector
- **OUTPUT_SIZE:** The maximum amount of characters to expect as the prediction output
- **BATCH_SIZE:** The amount of videos to read by batch
- **VAL_SPLIT:** The fraction between 0.0 and 1.0 of the videos to take as the validation set

Related to the standardization:
- **MEAN_R:** Arithmetic mean of the red channel in the training set
- **MEAN_G:** Arithmetic mean of the green channel in the training set
- **MEAN_B:** Arithmetic mean of the blue channel in the training set
- **STD_R:** Standard deviation of the red channel in the training set
- **STD_G:** Standard deviation of the green channel in the training set
- **STD_B:** Standard deviation of the blue channel in the training set

## To-do List

- [x] RGB standardization: Apply per-batch zero mean standardization
- [x] Augmentation: Make generators also output the horizontal flip of each video
- [x] Statistics: Record per-epoch statistics and other useful data visualizations.
- [ ] Documentation: Proper usage and code documentation
- [ ] Testing: Develop unit testing

## Built With

* [Python](https://www.python.org/) - The programming language
* [Keras](https://keras.io/) - The high-level neural network API

## Author

* **Omar Salinas** - [omarsalinas16](https://github.com/omarsalinas16) Developed as a bachelor's thesis @ UACJ - IIT

See also the list of [contributors](https://github.com/omarsalinas16/lipnet/contributors) who participated in this project.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details
