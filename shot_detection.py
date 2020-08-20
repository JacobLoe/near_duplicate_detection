import subprocess
import argparse
import os
from idmapper import TSVIdMapper
from tqdm import tqdm
import shutil

VERSION = '20200820'      # the version of the script
EXTRACTOR = 'shotdetection'


def shot_detect(v_path, f_path):
    keywords = ['-i', v_path,
                '-o', f_path,
                '-s', '60']
    process = ['shotdetect'] + keywords
    p = subprocess.run(process, bufsize=0,
                       shell=False,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)

    log_file = os.path.join(f_path, 'log.txt')
    with open(log_file, 'w') as f:
        f.write(str(p.stderr))


def main(videos_root, features_root, videoids, idmapper):
    # repeat until all movies are processed correctly
    for videoid in tqdm(videoids):
        try:
            video_rel_path = idmapper.get_filename(videoid)
        except KeyError as err:
            print("No such videoid: '{videoid}'".format(videoid=videoid))

        video_name = os.path.basename(video_rel_path)[:-4]
        features_dir = os.path.join(features_root, videoid, EXTRACTOR)
        if not os.path.isdir(features_dir):
            os.makedirs(features_dir)

        done_file_path = os.path.join(features_dir, '.done')

        v_path = os.path.join(videos_root, video_rel_path)
        if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == VERSION:
            print('shot detection results missing or version did not match, detecting shots for {video_name}'.format(video_name=video_name))

            # create the folder for the shot-detection, delete the old folder to prevent issues with older versions
            if not os.path.isdir(features_dir):
                os.makedirs(features_dir)
            else:
                shutil.rmtree()
                os.makedirs(features_dir)

            shot_detect(v_path, features_dir)

            # create a hidden file to signal that the asr for a movie is done
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
    args = parser.parse_args()

    idmapper = TSVIdMapper(args.file_mappings)
    videoids = args.videoids if len(args.videoids) > 0 else idmapper.get_ids()

    main(args.videos_dir, args.features_dir, videoids, idmapper)

