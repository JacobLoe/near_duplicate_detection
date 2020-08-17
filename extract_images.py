import os
import cv2
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET
import glob
import shutil
import numpy as np
from crop_image import trim
from PIL import Image
from video_aspect_ratio import get_aspect_ratios

FRAME_OFFSET_MS = 3*41  # frame offset in ms, one frame equals ~41ms, this jumps 3 frames ahead
TRIM_THRESHOLD = 12     # the threshold for the trim function, pixels with values lower are considered black and croppped
IMAGE_QUALITY = 90      # the quality to save images in, higher values mean less compression
VERSION = '20200425'      # the version of the script


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


def extract_images(v_path, f_path, file_extension, done, max_bbox_pro_shot, bounding_box_template, trim_frames, frame_width):
    video_name = os.path.split(v_path)[1][:-4]
    frames_dir = os.path.join(f_path, 'frames')

    done_file_path = os.path.join(frames_dir, '.done')

    # get the shot timestamps generated by shot-detect
    shot_timestamps = read_shotdetect_xml(os.path.join(f_path, 'shot_detection/result.xml'))

    a = cv2.VideoCapture(v_path).get(cv2.CAP_PROP_FPS)
    print(a)
    print(asda)

    # cut off the black borders of the movie
    if trim_frames == 'yes':
        if not os.path.isdir(frames_dir):
            print('extracting movie resolution for {}'.format(video_name))
            aux_bbox_dict = {}   # save the max resolution of each shot of a movie in a dict, keys are the start_frame of the shot
            for start_ms, end_ms in tqdm(shot_timestamps):
                # print('w/o offset', start_ms)
                # apply the offset to the timestamps
                start_ms = start_ms + FRAME_OFFSET_MS
                end_ms = end_ms - FRAME_OFFSET_MS
                frames_path = os.path.join(frames_dir, str(start_ms))
                aux_bbox_dict[start_ms] = get_trimmed_shot_resolution(v_path,
                                                                      frames_path,
                                                                      start_ms, end_ms,
                                                                      frame_width,
                                                                      file_extension)
            max_bbox_pro_shot[video_name] = aux_bbox_dict

            # get the lower and upper bounds of all the bounding boxes in a movie
            aux = [x for x in aux_bbox_dict.values()]
            lower_bounds = np.amin(aux, axis=0)
            upper_bounds = np.amax(aux, axis=0)
            bounding_box_template[video_name] = np.concatenate((lower_bounds[:2], upper_bounds[2:]))

            print('starting image extraction')
            for start_ms, end_ms in tqdm(shot_timestamps):

                # apply the offset to the timestamps
                start_ms = start_ms + FRAME_OFFSET_MS
                end_ms = end_ms - FRAME_OFFSET_MS

                # create a dir for a specific shot, the name are the boundaries in ms
                frames_path = os.path.join(f_path, 'frames', str(start_ms))

                # compare the resolution, after trimming of the shot, with the maximum resolution in the movie
                # and choose the larger resolution
                if (max_bbox_pro_shot[video_name][start_ms][2]-max_bbox_pro_shot[video_name][start_ms][0]) == 0:
                    max_bbox_pro_shot[video_name][start_ms] = bounding_box_template[video_name]

                crop_saved_frames(frames_path,
                                  start_ms, end_ms,
                                  max_bbox_pro_shot[video_name][start_ms],
                                  file_extension)

            # create a hidden file to signal that the image-extraction for a movie is done
            # write the current version of the script in the file
            with open(done_file_path, 'a') as d:
                d.write(VERSION)
            done += 1  # count the instances of the image-extraction done correctly
        # do nothing if a .done-file exists and the versions in the file and the script match
        elif os.path.isfile(done_file_path) and open(done_file_path, 'r').read() == VERSION:
            done += 1  # count the instances of the image-extraction done correctly
            print('image-extraction was already done for {}'.format(video_name))
        # if the folder already exists but the .done-file doesn't, delete the folder
        elif os.path.isfile(done_file_path) and not open(done_file_path, 'r').read() == VERSION:
            shutil.rmtree(frames_dir)
            print('versions did not match for {}'.format(video_name))
        elif not os.path.isfile(done_file_path):
            shutil.rmtree(frames_dir)
            print('image-extraction was not done correctly for {}'.format(video_name))

    else:
        # start extraction if no directory for the frames exists
        if not os.path.isdir(frames_dir):

            print('starting image extraction for {}'.format(video_name))
            for start_ms, end_ms in tqdm(shot_timestamps):

                # apply the offset to the timestamps
                start_ms = start_ms + FRAME_OFFSET_MS
                end_ms = end_ms - FRAME_OFFSET_MS

                # create a dir for a specific shot, the name are the boundaries in ms
                frames_path = os.path.join(frames_dir, str(start_ms))

                save_shot_frames(v_path,
                                 frames_path,
                                 start_ms, end_ms,
                                 frame_width,
                                 file_extension)
            # create a hidden file to signal that the image-extraction for a movie is done
            # write the current version of the script in the file
            with open(done_file_path, 'a') as d:
                d.write(VERSION)
            done += 1  # count the instances of the image-extraction done correctly
        # do nothing if a .done-file exists and the versions in the file and the script match
        elif os.path.isfile(done_file_path) and open(done_file_path, 'r').read() == VERSION:
            done += 1  # count the instances of the image-extraction done correctly
            print('image-extraction was already done for {}'.format(video_name))
        # if the folder already exists but the .done-file doesn't, delete the folder
        elif os.path.isfile(done_file_path) and not open(done_file_path, 'r').read() == VERSION:
            shutil.rmtree(frames_dir)
            print('versions did not match for {}'.format(video_name))
        elif not os.path.isfile(done_file_path):
            shutil.rmtree(frames_dir)
            print('image-extraction was not done correctly for {}'.format(video_name))

    return done, max_bbox_pro_shot, bounding_box_template


def main(videos_path, features_path, file_extension, trim_frames, frame_width):
    print('begin iterating through videos')

    list_videos_path = glob.glob(os.path.join(videos_path, '**/*.mp4'), recursive=True)  # get the list of videos in videos_dir

    if len(list_videos_path) > 1:
        cp = os.path.commonprefix(list_videos_path)  # get the common dir between paths found with glob
        list_features_path = [os.path.join(
                             os.path.join(features_path,
                             os.path.relpath(p, cp))[:-4])  # add a new dir 'VIDEO_FILE_NAME/shot_detection' to the path
                             for p in list_videos_path]  # create a list of paths where all the data (shot-detection,frames,features) are saved to

    else:
        # get the directory for the movie in the videos_dir
        r = os.path.relpath(list_videos_path[0], args.videos_dir)[:-4]
        list_features_path = [os.path.join(args.features_dir, r)]

    max_bbox_pro_shot = {}
    bounding_box_template = {}
    done = 0
    while done < len(list_features_path):  # repeat until all movies in the list have been processed correctly
        print('-------------------------------------------------------')
        for v_path, f_path in tqdm(zip(list_videos_path, list_features_path), total=len(list_videos_path)):
            done, max_res_pro_shot, bounding_box_template = extract_images(v_path, f_path, file_extension, done, max_bbox_pro_shot, bounding_box_template, trim_frames, frame_width)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored,"
                                           "the names of sub-directories need to start with different letters")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument("--trim_frames", default='no', choices=('yes', 'no'), help="decide whether to remove or keep black borders in the movies")
    parser.add_argument("--frame_width", type=int, help="set the width at which the frames are saved")
    parser.add_argument("--file_extension", default='.jpeg', choices=('.jpeg', '.png'), help="define the file-extension of the frames, only .png and .jpg are supported, default is .jpeg")
    args = parser.parse_args()

    main(args.videos_dir, args.features_dir, args.file_extension, args.trim_frames, args.frame_width)
