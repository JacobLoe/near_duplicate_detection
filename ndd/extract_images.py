import os
import cv2
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from utils import get_aspect_ratios, trim, read_shotdetect
import shutil
import logging

FRAME_OFFSET_MS = 3  # frame offset in frames, jumps 3 frames ahead
TRIM_THRESHOLD = 12     # the threshold for the trim function, pixels with values lower are considered black and croppped
IMAGE_QUALITY = 90      # the quality to save images in, higher values mean less compression
VERSION = '20210214'      # the date the script was last changed
EXTRACTOR = 'frames'
STANDALONE = True  # manages the creation of .done-files, if set to false no .done-files are created and the script will always overwrite old results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False    # prevent log messages from appearing twice


# read video file frame by frame, beginning and ending with a timestamp
def save_shot_frames(video_path, features_dir, start_ms, end_ms, frame_width, file_extension):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Unable to read from video: '{v_path}'".format(v_path=video_path))

    display_aspect_ratio, pixel_aspect_ratio, storage_aspect_ratio = get_aspect_ratios(video_path)
    if display_aspect_ratio == 0 or pixel_aspect_ratio == 0 or storage_aspect_ratio == 0:
        raise Exception('Something went wrong while extracting the aspect ratio for')

    # compute the offset depending on the fps of the video
    offset = FRAME_OFFSET_MS * int(1000/vid.get(cv2.CAP_PROP_FPS))
    start_ms = start_ms + offset
    end_ms = end_ms - offset

    for timestamp in range(start_ms, end_ms, 1000):

        vid.set(cv2.CAP_PROP_POS_MSEC, timestamp)
        ret, frame = vid.read()

        # check if the correct aspect ratio is used
        height, width, chan = frame.shape
        new_height, new_width = height, width * pixel_aspect_ratio
        frame = cv2.resize(frame, (int(new_width), int(new_height)))

        if frame_width:
            # resize the frame according to the frame_width provided and the aspect ratio of the frame
            resolution_old = np.shape(Image.fromarray(frame))
            ratio = resolution_old[1]/resolution_old[0]
            frame_height = int(frame_width/ratio)
            resolution_new = (frame_width, frame_height)
            frame = cv2.resize(frame, resolution_new)

        name = os.path.join(features_dir, (str(timestamp)+'.'+file_extension))
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if file_extension == "jpg":
           format='jpeg'
        elif file_extension == 'png':
           format='png'
        else:
           raise Exception("Unknown file format: {0}".format(file_extension))
        frame.save(name, format=format, quality=IMAGE_QUALITY)


def get_trimmed_shot_resolution(video_path, features_dir, start_ms, end_ms, frame_width, file_extension):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Unable to read from video: '{v_path}'".format(v_path=video_path))

    display_aspect_ratio, pixel_aspect_ratio, storage_aspect_ratio = get_aspect_ratios(video_path)
    if display_aspect_ratio == 0 or pixel_aspect_ratio == 0 or storage_aspect_ratio == 0:
        raise Exception('Something went wrong while extracting the aspect ratio')

    # compute the offset depending on the fps of the video
    offset = FRAME_OFFSET_MS * int(1000/vid.get(cv2.CAP_PROP_FPS))
    start_ms = start_ms + offset
    end_ms = end_ms - offset

    bounding_boxes = []
    for timestamp in range(start_ms, end_ms, 1000):

        vid.set(cv2.CAP_PROP_POS_MSEC, timestamp)
        ret, frame = vid.read()

        # check if the correct aspect ratio is used
        height, width, chan = frame.shape
        new_height, new_width = height, width * pixel_aspect_ratio
        frame = cv2.resize(frame, (int(new_width), int(new_height)))

        if frame_width:
            # resize the frame to according to the frame_width provided and the aspect ratio of the frame
            resolution_old = np.shape(Image.fromarray(frame))
            ratio = resolution_old[1]/resolution_old[0]
            frame_height = int(frame_width/ratio)
            resolution_new = (frame_width, frame_height)
            frame = cv2.resize(frame, resolution_new)

        # use the trim function and save the resulting resolution
        frame_array = Image.fromarray(frame)

        # get the trimmed frame and the corresponding bounding box
        # if none exist define both as empty
        try:
            frame_array, bounding_box = trim(frame_array, TRIM_THRESHOLD)
        except:
            bounding_box = (0, 0, np.shape(frame_array)[1], np.shape(frame_array)[0])
        bounding_boxes.append(bounding_box)

        # save the frame for later use
        name = os.path.join(features_dir, (str(timestamp) + '.' + file_extension))
        cv2.imwrite(name, frame)

    # if the shot is too short return a empty/invalid bounding box
    if not bounding_boxes:
        return np.array((0, 0, 0, 0)), offset   # FIXME find a better solution to deal with the emtpy shots

    # get the lower and upper bounds of all the bounding boxes in a shot
    lower_bounds = np.amin(bounding_boxes, axis=0)
    upper_bounds = np.amax(bounding_boxes, axis=0)
    shot_bounding_box = tuple(np.concatenate((lower_bounds[:2], upper_bounds[2:])))

    # return the specific offset for the video too, to later use it in the crop_saved_frames function
    return shot_bounding_box, offset


def crop_saved_frames(frames_path, start_ms, end_ms, bounding_box, file_extension):

    for timestamp in range(start_ms, end_ms, 1000):
        name = os.path.join(frames_path, (str(timestamp)+'.'+file_extension))

        frame = Image.open(name)
        frame = frame.crop(bounding_box)
        if file_extension == 'jpg':
           format='jpeg'
        elif file_extension == 'png':
           format='png'
        else:
           raise Exception("Unknown file extension: {0}".format(file_extension))
        frame.save(name, format=format, quality=IMAGE_QUALITY)


def main(features_root, file_extension, trim_frames, frame_width, videoids, force_run):
    frame_width = int(frame_width)
    max_bbox_pro_shot = {}
    bounding_box_template = {}
    # repeat until all movies are processed correctly
    for videoid in tqdm(videoids):

        features_dir = os.path.join(features_root, videoid, EXTRACTOR)

        v_path = os.path.join(features_root, videoid, 'media', videoid+'.mp4')
        # get the shot timestamps generated by shot-detect
        shot_timestamps = read_shotdetect(os.path.join(features_root, videoid, 'shotdetect', videoid+'.csv'))
        if not shot_timestamps:
            raise Exception('No shots could be found. Check the shotdetect results again')

        # check .done-file of the shotdetection for parameters to be added to the .done-file of the image extraction
        # assume that the shotdetection is set to STANDALONE if the image extraction is, Otherwise this check will be skipped
        previous_parameters = ''
        try:
            if STANDALONE:
                with open(os.path.join(features_root, videoid, 'shotdetect', '.done')) as done_file:
                    for i, line in enumerate(done_file):
                        if not i == 0:  # exclude the version of the previous script
                            previous_parameters += line
        except FileNotFoundError as err:
            raise Exception('The results of the shotdetection cannot be found. Shotdetection has to be run again')

        done_file_path = os.path.join(features_dir, '.done')
        # create the version for a run, based on the script version and the used parameters
        done_version = VERSION+'\n'+file_extension+'\n'+trim_frames+'\n'+str(frame_width)+'\n'+str(TRIM_THRESHOLD)+'\n'+str(IMAGE_QUALITY)+'\n'+previous_parameters

        if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == done_version or force_run == 'True':
            logger.info('image extraction results missing or version did not match, extracting images for')

            # create the folder for the frames, delete the old folder to prevent issues with older versions
            if not os.path.isdir(features_dir):
                os.makedirs(features_dir)
            else:
                shutil.rmtree(features_dir)
                os.makedirs(features_dir)

            if trim_frames == 'yes':

                logger.info('extracting movie resolution')
                aux_bbox_dict = {}  # save the max resolution of each shot of a movie in a dict, keys are the start_frame of the shot
                for start_ms, end_ms in tqdm(shot_timestamps):

                    bbox_dict, offset = get_trimmed_shot_resolution(v_path,
                                                                    features_dir,
                                                                    start_ms, end_ms,
                                                                    frame_width,
                                                                    file_extension)
                    aux_bbox_dict[start_ms+offset] = bbox_dict

                max_bbox_pro_shot[videoid] = aux_bbox_dict
                # get the lower and upper bounds of all the bounding boxes in a movie
                aux = [x for x in aux_bbox_dict.values()]
                lower_bounds = np.amin(aux, axis=0)
                upper_bounds = np.amax(aux, axis=0)
                bounding_box_template[videoid] = np.concatenate((lower_bounds[:2], upper_bounds[2:]))

                logger.info('starting image extraction')
                for start_ms, end_ms in tqdm(shot_timestamps):

                    # compute the offset depending on the fps of the video
                    start_ms = start_ms + offset
                    end_ms = end_ms - offset

                    # compare the resolution, after trimming of the shot, with the maximum resolution in the movie
                    # and choose the larger resolution
                    if (max_bbox_pro_shot[videoid][start_ms][2] - max_bbox_pro_shot[videoid][start_ms][0]) == 0:
                        max_bbox_pro_shot[videoid][start_ms] = bounding_box_template[videoid]

                    crop_saved_frames(features_dir,
                                      start_ms, end_ms,
                                      max_bbox_pro_shot[videoid][start_ms],
                                      file_extension)
            else:
                for start_ms, end_ms in tqdm(shot_timestamps):

                    save_shot_frames(v_path,
                                     features_dir,
                                     start_ms, end_ms,
                                     frame_width,
                                     file_extension)

            # create a hidden file to signal that the image extraction for a movie is done
            # write the current version of the script in the file
            if STANDALONE:
                with open(done_file_path, 'w') as d:
                    d.write(done_version)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("features_root", help="the directory where the images are to be stored")
    parser.add_argument("videoids", help="List of video ids. If empty, entire corpus is iterated.", nargs='*')
    parser.add_argument("--trim_frames", default='no', choices=('yes', 'no'), help="decide whether to remove or keep black borders in the movies")
    parser.add_argument("--frame_width", type=int, default=299, help="set the width at which the frames are saved")
    parser.add_argument("--file_extension", default='jpg', choices=('jpg', 'png'), help="define the file-extension of the frames, only .png and .jpg are supported, default is .jpg")
    parser.add_argument("--force_run", default='False', help='sets whether the script runs regardless of the version of .done-files')
    args = parser.parse_args()

    main(args.features_root, args.file_extension, args.trim_frames, args.frame_width, args.videoids, args.force_run)
