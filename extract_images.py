import os
import cv2
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET
import glob
import shutil
import time
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


def save_shot_frames(video_path, frame_path, start_ms, end_ms):
    frame_size = (299, 299)
    vid = cv2.VideoCapture(video_path)

    for i in range(int(end_ms/1000-start_ms/1000)+1):
        if not (start_ms/1000+i) == (int(end_ms/1000-start_ms/1000)+1):
            vid.set(cv2.CAP_PROP_POS_MSEC, start_ms+i*1000)
            ret, frame = vid.read()
            frame = cv2.resize(frame, frame_size)
            name = os.path.join(frame_path, (str(start_ms+i*1000)+'.png'))
            cv2.imwrite(name, frame)
#########################################################################################################


def main(videos_path, features_path):
    print('begin iterating through videos')

    list_videos_path = glob.glob(os.path.join(videos_path, '**/*.mp4'), recursive=True)  # get the list of videos in videos_dir

    cp = os.path.commonprefix(list_videos_path)  # get the common dir between paths found with glob

    list_features_path = [os.path.join(
                         os.path.join(features_path,
                         os.path.relpath(p, cp))[:-4])  # add a new dir 'VIDEO_FILE_NAME/shot_detection' to the path
                         for p in list_videos_path]  # create a list of paths where all the data (shot-detection,frames,features) are saved to

    done = 0
    while done < len(list_features_path):  # repeat until all movies in the list have been processed correctly
        print('-------------------------------------------------------')
        for v_path, f_path in tqdm(zip(list_videos_path, list_features_path), total=len(list_videos_path)):
            frames_dir = os.path.join(f_path, 'frames')
            if not os.path.isdir(frames_dir) and not os.path.isfile(os.path.join(frames_dir, '.done')):
                print('starting image extraction')
                # get the shot timestamps generated by shot-detect

                shot_timestamps = read_shotdetect_xml(os.path.join(f_path, 'shot_detection/result.xml'))

                for start_frame, end_frame in tqdm(shot_timestamps):
                    # create a dir for a specific shot, the name are the boundaries in ms
                    frames_path = os.path.join(f_path, 'frames', str(start_frame))
                    if not os.path.isdir(frames_path):
                        os.makedirs(frames_path)
                    save_shot_frames(v_path,
                                     frames_path,
                                     start_frame, end_frame)

                # create a hidden file to signal that the image-extraction for a movie is done
                open(os.path.join(frames_dir, '.done'), 'a').close()
                done += 1  # count the instances of the image-extraction done correctly
            elif os.path.isfile(os.path.join(frames_dir, '.done')):
                done += 1  # count the instances of the image-extraction done correctly
                print('image-extraction was already done for {}'.format(os.path.split(v_path)[1]))
            elif os.path.isdir(os.path.join(frames_dir)) and not os.path.isfile(os.path.join(frames_dir, '.done')):
                shutil.rmtree(f_path)
                print('image-extraction was not done correctly for {}'.format(os.path.split(v_path)[1]))

#########################################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    args = parser.parse_args()

    main(args.videos_dir, args.features_dir)
