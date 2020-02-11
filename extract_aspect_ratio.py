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


def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root = tree.getroot().findall('content')
    timestamps = []
    for child in root[0].iter():
        if child.tag == 'shot':
            items = child.items()
            attribs = child.attrib

            timestamps.append((int(attribs['msbegin']), int(attribs['msbegin']) + int(attribs['msduration']) - 1))  # ms
    return timestamps  # in ms


def save_aspect_ratio_to_csv(f_path, file_extension, done):

    shot_timestamps = read_shotdetect_xml(os.path.join(os.path.split(f_path)[0], 'shot_detection/result.xml'))

    # create the path for the .csv
    ar_dir_path = os.path.join(os.path.split(f_path)[0], 'aspect_ratio')

    if not os.path.isdir(ar_dir_path) and not os.path.isfile(os.path.join(ar_dir_path, '.done')):
        print('save aspect ratios for {}'.format(os.path.split(os.path.split(f_path)[0])[1]))
        if not os.path.isdir(ar_dir_path):
            os.makedirs(ar_dir_path)
        ar_csv_path = os.path.join(ar_dir_path, 'aspect_ratio_per_shot.csv')

        # define the list of possible aspect_ratios
        tu_ar_float = [21.01/9, 21/9, 16/9, 4/3, 1/1, 3/4, 9/16, 9/16.01]
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
        open(os.path.join(ar_dir_path, '.done'), 'a').close()
        done += 1
    elif os.path.isfile(os.path.join(ar_dir_path, '.done')):  # do nothing if a .done-file exists
        done += 1  # count the instances of the image-extraction done correctly
        print('image-extraction was already done for {}'.format(os.path.split(os.path.split(f_path)[0])[1]))
    # if the folder already exists but the .done-file doesn't, delete the folder
    elif os.path.isdir(os.path.join(ar_dir_path)) and not os.path.isfile(os.path.join(ar_dir_path, '.done')):
        shutil.rmtree(ar_dir_path)
        print('image-extraction was not done correctly for {}'.format(os.path.split(os.path.split(f_path)[0])[1]))
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
    parser.add_argument("--file_extension", default='.jpg', choices=('.jpg', '.png'), help="define the file-extension of the frames, only .png and .jpg are supported, default is .jpg")
    args = parser.parse_args()

    main(args.features_dir, args.file_extension)