import os
import argparse
from tqdm import tqdm
import glob
import shutil
import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean
import xml.etree.ElementTree as ET

FRAME_OFFSET_MS = 3*41  # frame offset in ms, one frame equals ~42ms, this jumps 3 frames ahead
VERSION = '20200425'      # the version of the script

def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root = tree.getroot().findall('content')
    timestamps = []
    for child in root[0].iter():
        if child.tag == 'shot':
            attribs = child.attrib

            timestamps.append((int(attribs['msbegin']), int(attribs['msbegin']) + int(attribs['msduration']) - 1))  # ms
    return timestamps  # in ms


def save_aspect_ratio_to_csv(f_path, file_extension, done):
    video_name = os.path.split(os.path.split(f_path)[0])[1]

    shot_timestamps = read_shotdetect_xml(os.path.join(os.path.split(f_path)[0], 'shot_detection/result.xml'))

    # create the path for the .csv
    ar_dir_path = os.path.join(os.path.split(f_path)[0], 'aspect_ratio')
    done_file_path = os.path.join(ar_dir_path, '.done')

    if not os.path.isdir(ar_dir_path):
        print('save aspect ratios for {}'.format(video_name))
        if not os.path.isdir(ar_dir_path):
            os.makedirs(ar_dir_path)

        ar_csv_path = os.path.join(ar_dir_path, 'aspect_ratio_per_shot_'+os.path.split(os.path.split(f_path)[0])[1]+'.csv')

        # define the list of possible aspect_ratios
        tu_ar_float = [2.55, 21./9., 16./9., 4./3., 1./1., 3./4., 9./16., 9./21.]
        tu_ar_str = ['>21/9', '21/9', '16/9', '4/3', '1/1', '3/4', '9/16', '<9/16']

        with open(ar_csv_path, 'w', newline='') as f:
            for start_ms, end_ms in tqdm(shot_timestamps):
                # apply the offset to the timestamps
                start_ms = start_ms + FRAME_OFFSET_MS
                end_ms = end_ms - FRAME_OFFSET_MS
                if not list(range(start_ms, end_ms, 1000)) == []:   # if the shot with the offset is too short, the shot is ignored

                    frame = str(start_ms)+file_extension
                    shot_path = os.path.join(f_path, str(start_ms), frame)
                    frame = Image.open(shot_path)

                    # convert the resolution of the shot to an aspect ratio
                    # set 0 if resolution is 0
                    if not np.shape(frame)[0] == 0:
                        ar = np.shape(frame)[1]/np.shape(frame)[0]
                    else:
                        ar = 0

                    # calculate the distance of the shot aspect ratio to the ratios in the list
                    dist = [euclidean(ar, x) for x in tu_ar_float]
                    line = str(start_ms-FRAME_OFFSET_MS)+' '+str(end_ms+FRAME_OFFSET_MS)+' '+str(tu_ar_str[np.argmin(dist)])    # save the shots without the offset
                    f.write(line)
                    f.write('\n')

        # create a hidden file to signal that the image-extraction for a movie is done
        # write the current version of the script in the file
        with open(done_file_path, 'a') as d:
            d.write(VERSION)
            done += 1  # count the instances of the image-extraction done correctly
            # do nothing if a .done-file exists and the versions in the file and the script match

    elif os.path.isfile(done_file_path) and open(done_file_path, 'r').read() == VERSION:
        done += 1  # count the instances of the image-extraction done correctly
        print('aspect-ratio-extraction was already done for {}'.format(video_name))
    # if the folder already exists but the .done-file doesn't, delete the folder
    elif os.path.isfile(done_file_path) and not open(done_file_path, 'r').read() == VERSION:
        shutil.rmtree(ar_dir_path)
        print('versions did not match for {}'.format(video_name))
    elif not os.path.isfile(done_file_path):
        shutil.rmtree(ar_dir_path)
        print('aspect-ratio-extraction was not done correctly for {}'.format(video_name))

    return done


def main(features_path, file_extension):
    print('begin iterating through videos')

    list_features_path = glob.glob(os.path.join(features_path, '**/frames'), recursive=True)

    done = 0
    while done < len(list_features_path):  # repeat until all movies in the list have been processed correctly
        print('-------------------------------------------------------')
        for f_path in tqdm(list_features_path, total=len(list_features_path)):
            done = save_aspect_ratio_to_csv(f_path, file_extension, done)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument("--file_extension", default='.jpeg', choices=('.jpeg', '.png'), help="define the file-extension of the frames, only .png and .jpg are supported, default is .jpg")
    args = parser.parse_args()

    main(args.features_dir, args.file_extension)