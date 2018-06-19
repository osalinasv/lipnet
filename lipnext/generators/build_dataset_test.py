import pickle
import os
import glob
import numpy as np
import argparse
import sys
sys.path.append("..\\..")
from common.files import read_subfolders


class Dataset(object):
    def __init__(self, data_path, absolute_max_string_len=30):
        self.data_path = data_path
        self.absolute_max_string_len = absolute_max_string_len
        self.train_path = os.path.join(self.data_path, 'train')
        self.val_path = os.path.join(self.data_path, 'val')
        self.align_path = os.path.join(self.data_path, 'align')

    def get_cache_path(self):
        return self.data_path.rstrip('/') + '.cache'

    def build_dataset(self):
        cache_path = self.get_cache_path()

        if os.path.isfile(self.get_cache_path()):
            print("\nLoading dataset list from cache...")
            with open (self.get_cache_path(), 'rb') as fp:
                self.train_list, self.val_list, self.align_hash = pickle.load(fp)
        else:
            print("\nEnumerating dataset list from disk...")
            #self.train_list = self.enumerate_videos(os.path.join(self.train_path, '*', '*'))
            self.train_list = read_subfolders(self.train_path)
            print("\nTrain: {}".format(self.train_list))
            self.val_list = read_subfolders(self.val_path)
            #self.val_list   = self.enumerate_videos(os.path.join(self.val_path, '*', '*'))
            #print("\nVal: {}".format(self.val_list))
            self.align_hash = self.enumerate_align_hash(self.train_list + self.val_list)
            with open(self.get_cache_path(), 'wb') as fp:
                pickle.dump((self.train_list, self.val_list, self.align_hash), fp)
        print(cache_path)
        print("Found {} videos for training.".format(self.training_size))
        print("Found {} videos for validation.".format(self.validation_size))
        print("")

    def enumerate_videos(self, path):
        video_list = []
        for video_path in glob.glob(path):

            video_list.append(video_path)
        return video_list

    def enumerate_align_hash(self, video_list):
        align_hash = {}
        for video_path in video_list:
            video_id = os.path.splitext(video_path)[0].split('\\')[-1]
            align_path = os.path.join(self.align_path, video_id)+".align"
            align_hash[video_id] = Align(self.absolute_max_string_len).from_file(align_path)
        return align_hash

class Align(object):
    def __init__(self, absolute_max_string_len=32):
        self.absolute_max_string_len = absolute_max_string_len

    def from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        align = [(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.strip().split(" ") for x in lines]]
        self.build(align)
        return self

    def build(self, align):
        self.align = self.strip(align, ['sp','sil'])
        self.sentence = self.get_sentence(align)
        self.label = self.get_label(self.sentence)
        self.padded_label = self.get_padded_label(self.label)

    def strip(self, align, items):
        return [sub for sub in align if sub[2] not in items]

    def get_sentence(self, align):
        return " ".join([y[-1] for y in align if y[-1] not in ['sp', 'sil']])

    def get_label(self, sentence):
        ret = []
        for char in sentence:
            if char >= 'a' and char <= 'z':
                ret.append(ord(char) - ord('a'))
            elif char == ' ':
                ret.append(26)
        return ret

    # Returns an array that is of size absolute_max_string_len. Fills the left spaces with -1 in case the len(label) is less than absolute_max_string_len.
    def get_padded_label(self, label):
        padding = np.ones((self.absolute_max_string_len-len(label))) * -1
        return np.concatenate((np.array(label), padding), axis=0)

if __name__ == '__main__':
    '''
    build_dataset_test.py
    Obtains the path to the validation and training folders    
    
    Usage:
        >>> python build_dataset_test.py -d [data_path] -m [max_string_len]
    
    Example:
        >>> python build_dataset_test.py -d ..\..\data
    
    '''
    # TODO: change instead of specify the train number an val number only specify the porcentage of training and have the option to clear the current folders
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data-path", required=True, help="Path to data")
    ap.add_argument("-m", "--max-string-len", required=False, help="(Optional) Max string lenght",type=int, default=32)
    args = vars(ap.parse_args())

    data_path = args["data_path"]
    max_string_len = args["max_string_len"]

    dataset = Dataset(data_path, 32).build_dataset()



