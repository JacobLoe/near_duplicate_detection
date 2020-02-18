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

FRAME_OFFSET_MS = 3*41  # frame offset in ms, one frame equals ~41ms, this jumps 3frames ahead
TRIM_THRESHOLD = 12     # defines the threshold
#################################################################


def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root = tree.getroot().findall('content')
    timestamps = []
    for child in root[0].iter():
        if child.tag == 'shot':
            attribs = child.attrib
               
            timestamps.append((int(attribs['msbegin']), int(attribs['msbegin'])+int(attribs['msduration'])-1))  # ms
    return timestamps  # in ms
#############################################################################################
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
        cv2.imwrite(name, frame)


def get_trimmed_shot_resolution(video_path, frames_path, start_ms, end_ms, frame_width, file_extension, trim_threshold):
    vid = cv2.VideoCapture(video_path)
    display_aspect_ratio, pixel_aspect_ratio, storage_aspect_ratio = get_aspect_ratios(video_path)

    shot_resolutions = []
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
            frame_array, bounding_box = trim(frame_array, trim_threshold)
        except:
            frame_array = ()
            bounding_box = ()
        bounding_boxes.append(bounding_box)
        shot_resolutions.append(np.shape(frame_array))

        # save the frame for later use
        name = os.path.join(frames_path, (str(timestamp) + file_extension))
        cv2.imwrite(name, frame)
    max_shot_resolution = sorted(zip(shot_resolutions, bounding_boxes), reverse=True)
    # if either the shot is empty or .. is ... assign the shot an empty resolution and a empty bounding box
    if max_shot_resolution == [] or max_shot_resolution[0] == ():
        max_shot_resolution = [((0, 0, 3), ())]

    # return the maximum shot resolution and the and the corresponding bounding box
    # reverse the order of the resolution to make it work with opencv
    return max_shot_resolution[0][0][:2][::-1], max_shot_resolution[0][1]


def crop_saved_frames(frames_path, start_ms, end_ms, bounding_box, file_extension):

    for timestamp in range(start_ms, end_ms, 1000):
        name = os.path.join(frames_path, (str(timestamp)+file_extension))

        frame = Image.open(name)
        frame = frame.crop(bounding_box)
        frame.save(name)

#########################################################################################################


def extract_images(v_path, f_path, file_extension, done, max_res_pro_shot, resolution_template, trim_frames, frame_width, trim_threshold):
    video_name = os.path.split(v_path)[1][:-4]
    frames_dir = os.path.join(f_path, 'frames')
    # get the shot timestamps generated by shot-detect
    shot_timestamps = read_shotdetect_xml(os.path.join(f_path, 'shot_detection/result.xml'))#[:59]
    # cut off the black borders of the movie
    if trim_frames == 'yes':
        if not os.path.isdir(frames_dir) and not os.path.isfile(os.path.join(frames_dir, '.done')):
            print('extracting movie resolution for {}'.format(os.path.split(v_path)[1]))
            aux_res_dict = {}   # save the max resolution of each shot of a movie in a dict, keys are the start_frame of the shot
            for start_ms, end_ms in tqdm(shot_timestamps):
                # print('w/o offset', start_ms)
                # apply the offset to the timestamps
                start_ms = start_ms + FRAME_OFFSET_MS
                end_ms = end_ms - FRAME_OFFSET_MS
                # print('w offset', start_ms)
                frames_path = os.path.join(f_path, 'frames', str(start_ms))
                # print('frames_path', frames_path)
                aux_res_dict[start_ms] = get_trimmed_shot_resolution(v_path,
                                                                    frames_path,
                                                                    start_ms, end_ms,
                                                                    frame_width,
                                                                    file_extension,
                                                                    trim_threshold)
            max_res_pro_shot[video_name] = aux_res_dict
            resolution_template[video_name] = sorted(aux_res_dict.values(), reverse=True)[0]
            print('starting image extraction')
            for start_ms, end_ms in tqdm(shot_timestamps):

                # apply the offset to the timestamps
                start_ms = start_ms + FRAME_OFFSET_MS
                end_ms = end_ms - FRAME_OFFSET_MS

                # create a dir for a specific shot, the name are the boundaries in ms
                frames_path = os.path.join(f_path, 'frames', str(start_ms))

                # compare the resolution, after trimming of the shot, with the maximum resolution in the movie
                # and choose the larger resolution
                if max_res_pro_shot[video_name][start_ms][0][0] == 0:
                    max_res_pro_shot[video_name][start_ms] = resolution_template[video_name]

                crop_saved_frames(frames_path,
                                        start_ms, end_ms,
                                        max_res_pro_shot[video_name][start_ms][1],
                                        file_extension)

            # create a hidden file to signal that the image-extraction for a movie is done
            open(os.path.join(frames_dir, '.done'), 'a').close()
            done += 1  # count the instances of the image-extraction done correctly
        elif os.path.isfile(os.path.join(frames_dir, '.done')):     # do nothing if a .done-file exists
            done += 1  # count the instances of the image-extraction done correctly
            print('image-extraction was already done for {}'.format(os.path.split(v_path)[1]))
        # if the folder already exists but the .done-file doesn't, delete the folder
        elif os.path.isdir(os.path.join(frames_dir)) and not os.path.isfile(os.path.join(frames_dir, '.done')):
            shutil.rmtree(frames_dir)
            print('image-extraction was not done correctly for {}'.format(os.path.split(v_path)[1]))

    else:
        if not os.path.isdir(frames_dir) and not os.path.isfile(os.path.join(frames_dir, '.done')):
            print('starting image extraction for {}'.format(os.path.split(v_path)[1]))
            for start_ms, end_ms in tqdm(shot_timestamps):

                # apply the offset to the timestamps
                start_ms = start_ms + FRAME_OFFSET_MS
                end_ms = end_ms - FRAME_OFFSET_MS

                # create a dir for a specific shot, the name are the boundaries in ms
                frames_path = os.path.join(f_path, 'frames', str(start_ms))

                save_shot_frames(v_path,
                                 frames_path,
                                 start_ms, end_ms,
                                 frame_width,
                                 file_extension)
            # create a hidden file to signal that the image-extraction for a movie is done
            open(os.path.join(frames_dir, '.done'), 'a').close()
            done += 1  # count the instances of the image-extraction done correctly
        elif os.path.isfile(os.path.join(frames_dir, '.done')):     # do nothing if a .done-file exists
            done += 1  # count the instances of the image-extraction done correctly
            print('image-extraction was already done for {}'.format(os.path.split(v_path)[1]))
        # if the folder already exists but the .done-file doesn't, delete the folder
        elif os.path.isdir(os.path.join(frames_dir)) and not os.path.isfile(os.path.join(frames_dir, '.done')):
            shutil.rmtree(frames_dir)
            print('image-extraction was not done correctly for {}'.format(os.path.split(v_path)[1]))
    return done, max_res_pro_shot, resolution_template


def main(videos_path, features_path, file_extension, trim_frames, frame_width, trim_threshold):
    print('begin iterating through videos')

    list_videos_path = glob.glob(os.path.join(videos_path, '**/*.mp4'), recursive=True)  # get the list of videos in videos_dir

    cp = os.path.commonprefix(list_videos_path)  # get the common dir between paths found with glob

    list_features_path = [os.path.join(
                         os.path.join(features_path,
                         os.path.relpath(p, cp))[:-4])  # add a new dir 'VIDEO_FILE_NAME/shot_detection' to the path
                         for p in list_videos_path]  # create a list of paths where all the data (shot-detection,frames,features) are saved to

    max_res_pro_shot = {}
    resolution_template = {}
    done = 0
    while done < len(list_features_path):  # repeat until all movies in the list have been processed correctly
        print('-------------------------------------------------------')
        for v_path, f_path in tqdm(zip(list_videos_path, list_features_path), total=len(list_videos_path)):
            done, max_res_pro_shot, resolution_template = extract_images(v_path, f_path, file_extension, done, max_res_pro_shot, resolution_template, trim_frames, frame_width, trim_threshold)
#########################################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument("--trim_frames", default='no', choices=('yes', 'no'), help="decide whether to remove or keep black borders in the movies")
    parser.add_argument("--trim_threshold", type=int, default='12', help="set the threshhold for the trim function")
    parser.add_argument("--frame_width", type=int, help="set the width at which the frames are saved")
    parser.add_argument("--file_extension", default='.jpg', choices=('.jpg', '.png'), help="define the file-extension of the frames, only .png and .jpg are supported, default is .jpg")
    args = parser.parse_args()

    main(args.videos_dir, args.features_dir, args.file_extension, args.trim_frames, args.frame_width, args.trim_threshold)
