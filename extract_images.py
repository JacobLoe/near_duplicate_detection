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
from scipy.spatial.distance import euclidean
#################################################################


def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root = tree.getroot().findall('content')
    timestamps = []
    for child in root[0].iter():
        if child.tag == 'shot':
            items = child.items()
            timestamps.append((int(items[4][1]), int(items[4][1])+int(items[2][1])-1))  # ms
    return timestamps  # in ms
#############################################################################################
# read video file frame by frame, beginning and ending with a timestamp
def save_shot_frames(video_path, frame_path, start_ms, end_ms, frame_width, file_extension):
    vid = cv2.VideoCapture(video_path)
    for i in range(int(end_ms/1000-start_ms/1000)+1):
        if not (start_ms/1000+i) == (int(end_ms/1000-start_ms/1000)+1):
            vid.set(cv2.CAP_PROP_POS_MSEC, start_ms+i*1000)
            ret, frame = vid.read()
            if frame_width:
                print(frame_width)
                # resize the frame to according to the frame_width provided and the aspect ratio of the frame
                resolution_old = np.shape(Image.fromarray(frame))
                ratio = resolution_old[1]/resolution_old[0]
                frame_height = int(frame_width/ratio)
                resolution_new = (frame_width, frame_height)
                frame = cv2.resize(frame, resolution_new)
            name = os.path.join(frame_path, (str(start_ms+i*1000)+file_extension))
            cv2.imwrite(name, frame)


def rescale_saved_frames(frame_path, start_ms, end_ms, frame_resolution, file_extension):
    for i in range(int(end_ms / 1000 - start_ms / 1000) + 1):
        if not (start_ms / 1000 + i) == (int(end_ms / 1000 - start_ms / 1000) + 1):
            name = os.path.join(frame_path, (str(start_ms+i*1000)+file_extension))
            frame = cv2.imread(name)
            frame = cv2.resize(frame, frame_resolution)
            cv2.imwrite(name, frame)


def get_trimmed_shot_resolution(video_path, frame_path, start_ms, end_ms, frame_width, file_extension):
    vid = cv2.VideoCapture(video_path)
    shot_resolutions = []
    for i in range(int(end_ms / 1000 - start_ms / 1000) + 1):
        if not (start_ms / 1000 + i) == (int(end_ms / 1000 - start_ms / 1000) + 1):
            vid.set(cv2.CAP_PROP_POS_MSEC, start_ms + i * 1000)
            ret, frame = vid.read()

            if frame_width:
                # resize the frame to according to the frame_width provided and the aspect ratio of the frame
                resolution_old = np.shape(Image.fromarray(frame))
                ratio = resolution_old[1]/resolution_old[0]
                frame_height = int(frame_width/ratio)
                resolution_new = (frame_width, frame_height)
                frame = cv2.resize(frame, resolution_new)

            # use the trim function and save the resulting resolution
            frame_array = Image.fromarray(frame)
            frame_array = trim(frame_array)
            shot_resolutions.append(np.shape(frame_array))

            # save the frame for later use
            name = os.path.join(frame_path, (str(start_ms + i * 1000) + file_extension))
            cv2.imwrite(name, frame)

    max_shot_resolution = sorted(shot_resolutions, reverse=True)
    if max_shot_resolution[0] == ():
        max_shot_resolution = [(0, 0, 3)]
    return max_shot_resolution[0][:2][::-1]     # reverse the order of the resolution to make it work with opencv


def save_aspect_ratio_in_csv(mrps, frames_dir, shot_timestamps):
    # print(mrps)
    # ar = [res[0]/(res[1]) if not res[1] == 0 else 0 for res in mrps.values()]
    tu_ar_float = [21.01/9, 21/9, 16/9, 4/3, 1/1, 3/4, 9/16, 9/16.01]
    tu_ar_str = ['>21/9', '21/9', '16/9', '4/3', '1/1', '3/4', '9/16', '<9/16']
    # [2.3344444444444448, 2.3333333333333335, 1.7777777777777777, 1.3333333333333333, 1.0, 0.75, 0.5625, 0.5621486570893192]
    ar_csv_path = os.path.join(frames_dir, 'aspect_ratio_per_shot.csv')
    with open(ar_csv_path, 'w', newline='') as f:
        for i, key in enumerate(mrps):
            # convert the resolution of the shot to an aspect ratio
            # set 0 if resolution is 0
            if not mrps[key][1] == 0:
                ar = mrps[key][0]/mrps[key][1]
            else:
                ar = 0

            # calculate the distance of the shot aspect ratio to the ratios in the list
            dist = [euclidean(ar, x) for x in tu_ar_float]
            # print(ar, mrps[key][0], mrps[key][1], dist, np.argmin(dist), tu_ar_str[np.argmin(dist)])
            line = str(key)+' '+str(shot_timestamps[i][1])+' '+str(tu_ar_str[np.argmin(dist)])
            f.write(line)
            f.write('\n')
#########################################################################################################


def extract_images(v_path, f_path, file_extension, done, max_res_pro_shot, resolution_template, trim_frames, frame_width):
    video_name = os.path.split(v_path)[1][:-4]
    frames_dir = os.path.join(f_path, 'frames')
    # get the shot timestamps generated by shot-detect
    shot_timestamps = read_shotdetect_xml(os.path.join(f_path, 'shot_detection/result.xml'))
    # cut of the black borders of the movie
    if trim_frames == 'yes':
        if not os.path.isdir(frames_dir) and not os.path.isfile(os.path.join(frames_dir, '.done')):
            print('extracting movie resolution ')
            aux_res_dict = {}   # save the max resolution of each shot of a movie in a dict, keys are the start_frame of the shot
            for start_frame, end_frame in tqdm(shot_timestamps):
                frames_path = os.path.join(f_path, 'frames', str(start_frame))
                if not os.path.isdir(frames_path):
                    os.makedirs(frames_path)
                aux_res_dict[start_frame] = get_trimmed_shot_resolution(v_path,
                                                                        frames_path,
                                                                        start_frame, end_frame,
                                                                        frame_width,
                                                                        file_extension)
            max_res_pro_shot[video_name] = aux_res_dict
            resolution_template[video_name] = sorted(aux_res_dict.values(), reverse=True)[0]

            print('starting image extraction')
            for start_frame, end_frame in tqdm(shot_timestamps):
                # create a dir for a specific shot, the name are the boundaries in ms
                frames_path = os.path.join(f_path, 'frames', str(start_frame))
                # compare the resolution, after trimming of the shot, with the maximum resolution in the movie
                # and choose the larger resolution
                if max_res_pro_shot[video_name][start_frame][0] < resolution_template[video_name][0]:
                    max_res_pro_shot[video_name][start_frame] = resolution_template[video_name]

                rescale_saved_frames(frames_path,
                                     start_frame, end_frame,
                                     max_res_pro_shot[video_name][start_frame],
                                     file_extension)

            # save the aspect ratio of the shot in a .csv-file
            save_aspect_ratio_in_csv(max_res_pro_shot[video_name], frames_dir, shot_timestamps)

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
            print('starting image extraction')
            for start_frame, end_frame in tqdm(shot_timestamps):
                # create a dir for a specific shot, the name are the boundaries in ms
                frames_path = os.path.join(f_path, 'frames', str(start_frame))
                if not os.path.isdir(frames_path):
                    os.makedirs(frames_path)
                save_shot_frames(v_path,
                                 frames_path,
                                 start_frame, end_frame,
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


def main(videos_path, features_path, file_extension, trim_frames, frame_width):
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
            done, max_res_pro_shot, resolution_template = extract_images(v_path, f_path, file_extension, done, max_res_pro_shot, resolution_template, trim_frames, frame_width)
#########################################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument("--trim_frames", default='no', choices=('yes', 'no'), help="decide whether to remove or keep black borders in the movies")
    parser.add_argument("--frame_width", type=int, help="set the width at which the frames are saved")
    parser.add_argument("--file_extension", default='.jpg', choices=('.jpg', '.png'), help="define the file-extension of the frames, only .png and .jpg are supported, default is .jpg")
    args = parser.parse_args()

    main(args.videos_dir, args.features_dir, args.file_extension, args.trim_frames, args.frame_width)
