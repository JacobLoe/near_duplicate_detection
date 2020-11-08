import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean
import xml.etree.ElementTree as ET
from idmapper import TSVIdMapper
import shutil
import cv2

FRAME_OFFSET_MS = 3  # frame offset in ms, one frame equals ~42ms, this jumps 3 frames ahead
VERSION = '20200910'      # the version of the script
EXTRACTOR = 'aspectratio'
STANDALONE = True  # manages the creation of .done-files, if set to false no .done-files are created and the script will always overwrite old results


def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root = tree.getroot().findall('content')
    timestamps = []
    for child in root[0].iter():
        if child.tag == 'shot':
            attribs = child.attrib

            timestamps.append((int(attribs['msbegin']), int(attribs['msbegin']) + int(attribs['msduration']) - 1))
    return timestamps


def save_aspect_ratio_to_csv(video_path, ar_dir_path, file_extension, videoname):
    # open the video and get the fps
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Unable to read from video: '{v_path}'".format(v_path=video_path))
    # compute the offset depending on the fps of the video
    offset = FRAME_OFFSET_MS * int(1000/vid.get(cv2.CAP_PROP_FPS))
    vid.release()
    cv2.destroyAllWindows()

    shot_timestamps = read_shotdetect_xml(os.path.join(os.path.split(ar_dir_path)[0], 'shotdetection/result.xml'))
    if not shot_timestamps:
        raise Exception('No shots could be found. Check the shotdetection results for "{video_name}" again'.format(video_name=videoname))

    ar_csv_path = os.path.join(ar_dir_path, 'aspect_ratio_per_shot_'+str(videoname)+'.csv')

    f_path = os.path.join(os.path.split(ar_dir_path)[0], 'frames')

    # define the list of possible aspect_ratios
    tu_ar_float = [2.55, 21./9., 16./9., 4./3., 1./1., 3./4., 9./16., 9./21.]
    tu_ar_str = ['>21/9', '21/9', '16/9', '4/3', '1/1', '3/4', '9/16', '<9/16']

    with open(ar_csv_path, 'w', newline='') as f:
        for start_ms, end_ms in tqdm(shot_timestamps):
            # apply the offset to the timestamps
            start_ms = start_ms + offset
            end_ms = end_ms - offset
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


def main(videos_root, features_root, file_extension, videoids, idmapper):
    # repeat until all movies are processed correctly
    for videoid in tqdm(videoids):
        try:
            video_rel_path = idmapper.get_filename(videoid)
        except KeyError as err:
            raise Exception("No such videoid: '{videoid}'".format(videoid=videoid))

        video_name = os.path.basename(video_rel_path)[:-4]
        features_dir = os.path.join(features_root, videoid, EXTRACTOR)
        v_path = os.path.join(videos_root, video_rel_path)

        # check .done-file of the image extraction for parameters to be added to the .done-file of the aspect ratio extraction
        # assume that the image extraction is set to STANDALONE if the aspect ratio extraction is, Otherwise this check will be skipped
        previous_parameters = ''
        try:
            if not force_run:
                with open(os.path.join(features_root, videoid, 'frames', '.done')) as done_file:
                    for i, line in enumerate(done_file):
                        if not i == 0:
                            previous_parameters = '\n' + previous_parameters + line
        except FileNotFoundError as err:
            raise Exception('The results of the image extraction for "{video_name}" cannot be found. The image extraction has to be run again'.format(video_name=video_name))

        done_file_path = os.path.join(features_dir, '.done')
        # create the version for a run, based on the script version and the used parameters
        done_version = VERSION+'\n'+file_extension+previous_parameters

        # get the shot timestamps generated by shot-detect
        if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == done_version or force_run:
            print('aspect ratio extraction results missing or version did not match, extracting aspect ratios for "{video_name}"'.format(video_name=video_name))
            # create the folder for the features, delete the old folder to prevent issues with older versions
            if not os.path.isdir(features_dir):
                os.makedirs(features_dir)
            else:
                shutil.rmtree(features_dir)
                os.makedirs(features_dir)

            save_aspect_ratio_to_csv(v_path, features_dir, file_extension, video_name)

            # create a hidden file to signal that the feature extraction for a movie is done
            # write the current version of the script in the file
            if STANDALONE:
                with open(done_file_path, 'w') as d:
                    d.write(done_version)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored,"
                                           "the names of sub-directories need to start with different letters")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument('file_mappings', help='path to file mappings .tsv-file')
    parser.add_argument("videoids", help="List of video ids. If empty, entire corpus is iterated.", nargs='*')
    parser.add_argument("--file_extension", default='.jpeg', choices=('.jpeg', '.png'), help="define the file-extension of the frames, only .png and .jpg are supported, default is .jpg")
    parser.add_argument("--force_run", default=False, type=bool, help='sets whether the script runs regardless of the version of .done-files')
    args = parser.parse_args()

    force_run = args.force_run

    idmapper = TSVIdMapper(args.file_mappings)
    videoids = args.videoids if len(args.videoids) > 0 else parser.error('no videoids found')

    main(args.videos_dir, args.features_dir, args.file_extension, videoids, idmapper)
