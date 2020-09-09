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

FRAME_OFFSET_MS = 3*41  # frame offset in ms, one frame equals ~41ms, this jumps 3 frames ahead
TRIM_THRESHOLD = 12     # the threshold for the trim function, pixels with values lower are considered black and croppped
IMAGE_QUALITY = 90      # the quality to save images in, higher values mean less compression
VERSION = '20200909'      # the version of the script
EXTRACTOR = 'frames'


def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root = tree.getroot().findall('content')
    timestamps = []
    for child in root[0].iter():
        if child.tag == 'shot':
            attribs = child.attrib
            timestamps.append((int(attribs['msbegin']), int(attribs['msbegin'])+int(attribs['msduration'])-1))  # ms
    return timestamps  # in ms


# read video file frame by frame, beginning and ending with a timestamp
def save_shot_frames(video_path, frames_path, start_ms, end_ms, frame_width, file_extension):
    vid = cv2.VideoCapture(video_path)
    display_aspect_ratio, pixel_aspect_ratio, storage_aspect_ratio = get_aspect_ratios(video_path)

    for timestamp in range(start_ms, end_ms, 1000):
        if not os.path.isdir(frames_path):
            os.makedirs(frames_path)
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


def get_trimmed_shot_resolution(video_path, frames_path, start_ms, end_ms, frame_width, file_extension):
    vid = cv2.VideoCapture(video_path)
    display_aspect_ratio, pixel_aspect_ratio, storage_aspect_ratio = get_aspect_ratios(video_path)

    bounding_boxes = []
    for timestamp in range(start_ms, end_ms, 1000):
        if not os.path.isdir(frames_path):
            os.makedirs(frames_path)
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
        # if none exist define both as empty sets
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
        return np.array((0, 0, 0, 0))   # FIXME find a better solution to deal with the emtpy shots

    # get the lower and upper bounds of all the bounding boxes in a shot
    lower_bounds = np.amin(bounding_boxes, axis=0)
    upper_bounds = np.amax(bounding_boxes, axis=0)
    shot_bounding_box = tuple(np.concatenate((lower_bounds[:2], upper_bounds[2:])))

    return shot_bounding_box


def crop_saved_frames(frames_path, start_ms, end_ms, bounding_box, file_extension):

    for timestamp in range(start_ms, end_ms, 1000):
        name = os.path.join(frames_path, (str(timestamp)+file_extension))

        frame = Image.open(name)
        frame = frame.crop(bounding_box)
        frame.save(name, format=file_extension[1:], quality=IMAGE_QUALITY)


def main(videos_root, features_root, file_extension, trim_frames, frame_width, videoids, idmapper):
    max_bbox_pro_shot = {}
    bounding_box_template = {}
    # repeat until all movies are processed correctly
    for videoid in tqdm(videoids):
        try:
            video_rel_path = idmapper.get_filename(videoid)
        except KeyError as err:
            print("No such videoid: '{videoid}'".format(videoid=videoid))

        video_name = os.path.basename(video_rel_path)[:-4]
        features_dir = os.path.join(features_root, videoid, EXTRACTOR)

        done_file_path = os.path.join(features_dir, '.done')

        v_path = os.path.join(videos_root, video_rel_path)
        # get the shot timestamps generated by shot-detect
        shot_timestamps = read_shotdetect_xml(os.path.join(features_root, videoid, 'shotdetection/result.xml'))
        if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == VERSION:
            print('image extraction results missing or version did not match, extracting images for {video_name}'.format(video_name=video_name))

            # create the folder for the frames, delete the old folder to prevent issues with older versions
            if not os.path.isdir(features_dir):
                os.makedirs(features_dir)
            else:
                shutil.rmtree(features_dir)
                os.makedirs(features_dir)
            #
            # if trim_frames == 'yes':
            #
            #     print('extracting movie resolution for {}'.format(video_name))
            #     aux_bbox_dict = {}  # save the max resolution of each shot of a movie in a dict, keys are the start_frame of the shot
            #     for start_ms, end_ms in tqdm(shot_timestamps):
            #         # apply the offset to the timestamps
            #         start_ms = start_ms + FRAME_OFFSET_MS
            #         end_ms = end_ms - FRAME_OFFSET_MS
            #         frames_path = os.path.join(features_dir, str(start_ms))
            #         aux_bbox_dict[start_ms] = get_trimmed_shot_resolution(v_path,
            #                                                               frames_path,
            #                                                               start_ms, end_ms,
            #                                                               frame_width,
            #                                                               file_extension)
            #     max_bbox_pro_shot[video_name] = aux_bbox_dict
            #     # get the lower and upper bounds of all the bounding boxes in a movie
            #     aux = [x for x in aux_bbox_dict.values()]
            #     lower_bounds = np.amin(aux, axis=0)
            #     upper_bounds = np.amax(aux, axis=0)
            #     bounding_box_template[video_name] = np.concatenate((lower_bounds[:2], upper_bounds[2:]))
            #
            #     print('starting image extraction')
            #     for start_ms, end_ms in tqdm(shot_timestamps):
            #
            #         # apply the offset to the timestamps
            #         start_ms = start_ms + FRAME_OFFSET_MS
            #         end_ms = end_ms - FRAME_OFFSET_MS
            #
            #         # create a dir for a specific shot, the name are the boundaries in ms
            #         frames_path = os.path.join(features_dir, str(start_ms))
            #
            #         # compare the resolution, after trimming of the shot, with the maximum resolution in the movie
            #         # and choose the larger resolution
            #         if (max_bbox_pro_shot[video_name][start_ms][2] - max_bbox_pro_shot[video_name][start_ms][0]) == 0:
            #             max_bbox_pro_shot[video_name][start_ms] = bounding_box_template[video_name]
            #
            #         crop_saved_frames(frames_path,
            #                           start_ms, end_ms,
            #                           max_bbox_pro_shot[video_name][start_ms],
            #                           file_extension)
            # else:
            #     for start_ms, end_ms in tqdm(shot_timestamps):
            #         # apply the offset to the timestamps
            #         start_ms = start_ms + FRAME_OFFSET_MS
            #         end_ms = end_ms - FRAME_OFFSET_MS
            #
            #         # create a dir for a specific shot, the name are the boundaries in ms
            #         frames_path = os.path.join(features_dir, str(start_ms))
            #
            #         save_shot_frames(v_path,
            #                          frames_path,
            #                          start_ms, end_ms,
            #                          frame_width,
            #                          file_extension)

            # create a hidden file to signal that the image extraction for a movie is done
            # write the current version of the script in the file
            with open(done_file_path, 'w') as d:
                d.write(VERSION)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored,"
                                           "the names of sub-directories need to start with different letters")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument('file_mappings', help='path to file mappings .tsv-file')
    parser.add_argument("videoids", help="List of video ids. If empty, entire corpus is iterated.", nargs='*')
    parser.add_argument("--trim_frames", default='no', choices=('yes', 'no'), help="decide whether to remove or keep black borders in the movies")
    parser.add_argument("--frame_width", type=int, help="set the width at which the frames are saved")
    parser.add_argument("--file_extension", default='.jpeg', choices=('.jpeg', '.png'), help="define the file-extension of the frames, only .png and .jpg are supported, default is .jpeg")
    args = parser.parse_args()

    idmapper = TSVIdMapper(args.file_mappings)
    videoids = args.videoids if len(args.videoids) > 0 else parser.error('no videoids found')

    main(args.videos_dir, args.features_dir, args.file_extension, args.trim_frames, args.frame_width, videoids, idmapper)
