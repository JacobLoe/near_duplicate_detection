import os
import cv2
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET
import numpy as np
from crop_image import trim
from PIL import Image
from video_aspect_ratio import get_aspect_ratios
from idmapper import TSVIdMapper
import shutil

FRAME_OFFSET_MS = 3  # frame offset in frames, jumps 3 frames ahead
TRIM_THRESHOLD = 12     # the threshold for the trim function, pixels with values lower are considered black and croppped
IMAGE_QUALITY = 90      # the quality to save images in, higher values mean less compression
VERSION = '20201026'      # the version of the script
EXTRACTOR = 'frames'
STANDALONE = True  # manages the creation of .done-files, if set to false no .done-files are created and the script will always overwrite old results


def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root = tree.getroot().findall('content')
    timestamps = []
    for child in root[0].iter():
        if child.tag == 'shot':
            attribs = child.attrib
            timestamps.append((int(attribs['msbegin']), int(attribs['msbegin'])+int(attribs['msduration'])-1))
    return timestamps


# read video file frame by frame, beginning and ending with a timestamp
def save_shot_frames(video_path, features_dir, start_ms, end_ms, frame_width, file_extension, video_name):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Unable to read from video: '{v_path}'".format(v_path=video_path))

    display_aspect_ratio, pixel_aspect_ratio, storage_aspect_ratio = get_aspect_ratios(video_path)
    if display_aspect_ratio == 0 or pixel_aspect_ratio == 0 or storage_aspect_ratio == 0:
        raise Exception('Something went wrong while extracting the aspect ratio for "{video_name}"'.format(video_name))

    # compute the offset depending on the fps of the video
    offset = FRAME_OFFSET_MS * int(1000/vid.get(cv2.CAP_PROP_FPS))
    start_ms = start_ms + offset
    end_ms = end_ms - offset

    # create a folder for a shot named after the timestamp of the first frame in that shot
    frames_path = os.path.join(features_dir, str(start_ms))
    if not os.path.isdir(frames_path):
        os.makedirs(frames_path)

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

        name = os.path.join(frames_path, (str(timestamp)+file_extension))
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame.save(name, format=file_extension[1:], quality=IMAGE_QUALITY)


def get_trimmed_shot_resolution(video_path, features_dir, start_ms, end_ms, frame_width, file_extension, video_name):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Unable to read from video: '{v_path}'".format(v_path=video_path))

    display_aspect_ratio, pixel_aspect_ratio, storage_aspect_ratio = get_aspect_ratios(video_path)
    if display_aspect_ratio == 0 or pixel_aspect_ratio == 0 or storage_aspect_ratio == 0:
        raise Exception('Something went wrong while extracting the aspect ratio for "{video_name}"'.format(video_name))

    # compute the offset depending on the fps of the video
    offset = FRAME_OFFSET_MS * int(1000/vid.get(cv2.CAP_PROP_FPS))
    start_ms = start_ms + offset
    end_ms = end_ms - offset

    # create a folder for a shot named after the timestamp of the first frame in that shot
    frames_path = os.path.join(features_dir, str(start_ms))
    if not os.path.isdir(frames_path):
        os.makedirs(frames_path)

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
        name = os.path.join(frames_path, (str(timestamp) + file_extension))
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
        name = os.path.join(frames_path, (str(timestamp)+file_extension))

        frame = Image.open(name)
        frame = frame.crop(bounding_box)
        frame.save(name, format=file_extension[1:], quality=IMAGE_QUALITY)


def main(videos_root, features_root, file_extension, trim_frames, frame_width, videoids, idmapper):
    frame_width = int(frame_width)
    max_bbox_pro_shot = {}
    bounding_box_template = {}
    # repeat until all movies are processed correctly
    for videoid in tqdm(videoids):
        try:
            video_rel_path = idmapper.get_filename(videoid)
        except KeyError as err:
            raise KeyError("No such videoid: '{videoid}'".format(videoid=videoid))

        video_name = os.path.basename(video_rel_path)[:-4]
        features_dir = os.path.join(features_root, videoid, EXTRACTOR)

        v_path = os.path.join(videos_root, video_rel_path)
        # get the shot timestamps generated by shot-detect
        shot_timestamps = read_shotdetect_xml(os.path.join(features_root, videoid, 'shotdetection/result.xml'))
        if not shot_timestamps:
            raise Exception('No shots could be found. Check the shotdetection results for "{video_name}" again'.format(video_name=video_name))

        # check .done-file of the shotdetection for parameters to be added to the .done-file of the image extraction
        # assume that the shotdetection is set to STANDALONE if the image extraction is, Otherwise this check will be skipped
        previous_parameters = ''
        try:
            if STANDALONE:
                with open(os.path.join(features_root, videoid, 'shotdetection', '.done')) as done_file:
                    for i, line in enumerate(done_file):
                        if not i == 0:
                            previous_parameters = '\n' + previous_parameters + line
        except FileNotFoundError as err:
            raise Exception('The results of the shotdetection for "{video_name}" cannot be found. Shotdetection has to be run again'.format(video_name=video_name))

        done_file_path = os.path.join(features_dir, '.done')
        # create the version for a run, based on the script version and the used parameters
        done_version = VERSION+'\n'+file_extension+'\n'+trim_frames+'\n'+str(frame_width)+'\n'+str(TRIM_THRESHOLD)+'\n'+str(IMAGE_QUALITY)+previous_parameters

        if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == done_version or force_run:
            print('image extraction results missing or version did not match, extracting images for "{video_name}"'.format(video_name=video_name))

            # create the folder for the frames, delete the old folder to prevent issues with older versions
            if not os.path.isdir(features_dir):
                os.makedirs(features_dir)
            else:
                shutil.rmtree(features_dir)
                os.makedirs(features_dir)

            if trim_frames == 'yes':

                print('extracting movie resolution for "{}"'.format(video_name))
                aux_bbox_dict = {}  # save the max resolution of each shot of a movie in a dict, keys are the start_frame of the shot
                for start_ms, end_ms in tqdm(shot_timestamps):

                    bbox_dict, offset = get_trimmed_shot_resolution(v_path,
                                                                    features_dir,
                                                                    start_ms, end_ms,
                                                                    frame_width,
                                                                    file_extension,
                                                                    video_name)
                    aux_bbox_dict[start_ms+offset] = bbox_dict

                max_bbox_pro_shot[video_name] = aux_bbox_dict
                # get the lower and upper bounds of all the bounding boxes in a movie
                aux = [x for x in aux_bbox_dict.values()]
                lower_bounds = np.amin(aux, axis=0)
                upper_bounds = np.amax(aux, axis=0)
                bounding_box_template[video_name] = np.concatenate((lower_bounds[:2], upper_bounds[2:]))

                print('starting image extraction')
                for start_ms, end_ms in tqdm(shot_timestamps):

                    # compute the offset depending on the fps of the video
                    start_ms = start_ms + offset
                    end_ms = end_ms - offset

                    frames_path = os.path.join(features_dir, str(start_ms))

                    # compare the resolution, after trimming of the shot, with the maximum resolution in the movie
                    # and choose the larger resolution
                    if (max_bbox_pro_shot[video_name][start_ms][2] - max_bbox_pro_shot[video_name][start_ms][0]) == 0:
                        max_bbox_pro_shot[video_name][start_ms] = bounding_box_template[video_name]

                    crop_saved_frames(frames_path,
                                      start_ms, end_ms,
                                      max_bbox_pro_shot[video_name][start_ms],
                                      file_extension)
            else:
                for start_ms, end_ms in tqdm(shot_timestamps):

                    save_shot_frames(v_path,
                                     features_dir,
                                     start_ms, end_ms,
                                     frame_width,
                                     file_extension,
                                     video_name)

            # create a hidden file to signal that the image extraction for a movie is done
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
    parser.add_argument("--trim_frames", default='no', choices=('yes', 'no'), help="decide whether to remove or keep black borders in the movies")
    parser.add_argument("--frame_width", type=int, default=299, help="set the width at which the frames are saved")
    parser.add_argument("--file_extension", default='.jpeg', choices=('.jpeg', '.png'), help="define the file-extension of the frames, only .png and .jpg are supported, default is .jpeg")
    parser.add_argument("--force_run", default=False, type=bool, help='sets whether the script runs regardless of the version of .done-files')
    args = parser.parse_args()

    force_run = args.force_run

    idmapper = TSVIdMapper(args.file_mappings)
    videoids = args.videoids if len(args.videoids) > 0 else parser.error('no videoids found')

    main(args.videos_dir, args.features_dir, args.file_extension, args.trim_frames, args.frame_width, videoids, idmapper)
