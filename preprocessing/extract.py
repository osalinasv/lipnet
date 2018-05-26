import fnmatch
import os
import sys

from preprocessing.extractor.extract_roi_frames import video_to_frames


def make_dir(path: str):
    if not os.path.exists(path) or not os.path.isdir(path):
        os.makedirs(path)


def find_files(path: str, pattern: str):
    for root, _, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.realpath(os.path.join(root, basename))
                yield filename


def extract(videos_path: str, pattern: str, output_path: str, predictor_path: str):
    """
    Extracts the frames in all videos inside videos_path that match pattern

    Usage:
        python extract.py [videos_path] [pattern] [output_path]

        videos_path         Path to videos directory
        pattern             Filename pattern to match
        output_path         Path for the extracted frames

    Example:
        python extract.py data/dataset *.mpg data/target data/predictors/shape_predictor_68_face_landmarks.dat

    :param videos_path:
    :param pattern:
    :param output_path:
    :param predictor_path:
    :return:
    """

    videos_path = os.path.realpath(videos_path)
    output_path = os.path.realpath(output_path)
    predictor_path = os.path.realpath(predictor_path)

    print('\nEXTRACT\n')
    print('Searching for files in: {}\nMatch for: {}'.format(videos_path, pattern))

    for file_path in find_files(videos_path, pattern):
        group_dir = os.path.basename(os.path.dirname(file_path))
        video_dir = os.path.splitext(os.path.basename(file_path))[0]

        video_full_dir = os.path.join(group_dir, video_dir)

        vid_cutouts_target_dir = os.path.join(output_path, video_full_dir)
        make_dir(vid_cutouts_target_dir)

        video_to_frames(file_path, vid_cutouts_target_dir, predictor_path)

    print('Finished extraction successfully\n')


if __name__ == '__main__':
    argv_len = len(sys.argv)

    if argv_len < 3 or argv_len > 5:
        print('''
    extract.py
        Extracts the frames in all videos inside videos_path that match pattern
    
    Usage:
        python extract.py [videos_path] [pattern] [output_path] [predictor_path]
        
        videos_path         Path to videos directory
        pattern             (Optional) Filename pattern to match
        output_path         Path for the extracted frames
        predictor_path      (Optional) Path to the predictor .dat file

    Example:
        python extract.py data/dataset *.mpg data/target data/predictors/shape_predictor_68_face_landmarks.dat

''')
        exit()

    i_path = None
    o_path = None
    pat = '*.mpg'
    p_path = os.path.join(__file__, '..', '..', 'data', 'predictors', 'shape_predictor_68_face_landmarks.dat')

    if argv_len == 3:
        i_path = sys.argv[1]
        o_path = sys.argv[2]

    if argv_len >= 4:
        i_path = sys.argv[1]
        pat = sys.argv[2]
        o_path = sys.argv[3]

    if argv_len == 5:
        p_path = sys.argv[4]

    if i_path is None or o_path is None:
        print('Both input and output are required\n')
        exit()

    extract(i_path, pat, o_path, p_path)
