import subprocess
import argparse
import os
from idmapper import TSVIdMapper
from tqdm import tqdm
import shutil
import xml.etree.ElementTree as ET

VERSION = '20200910'      # the version of the script
EXTRACTOR = 'shotdetection'     #
STANDALONE = False  # manages the creation of .done-files, if set to false no .done-files are created and the script will always overwrite old results


def check_shotdetection(xml_path, video_name, stderr, stdout):
    # checks if the shotdetection has run correctly
    # read the result.xml from a shotdetection
    tree = ET.parse(xml_path)
    root = tree.getroot().findall('content')
    c = []
    for child in root[0].iter():
        if child.tag == 'video' or child.tag == 'audio':
            c.append(child.text)
    # check the content of video and audio.
    if c[0] is None or c[0] == 'null' and c[1] is None or c[1] == 'null':
        raise Exception('{videoname} is not a valid video. Either the file is corrupted or not a video \n'
                        'Shotdetection error: {stderr} \n'
                        'Shotdetection output: {stdout}'.format(videoname=video_name, stderr=stderr, stdout=stdout))


def shot_detect(v_path, f_path, sensitivity, video_name):
    keywords = ['-i', v_path,
                '-o', f_path,
                '-s', sensitivity]
    process = ['shotdetect'] + keywords
    p = subprocess.run(process, bufsize=0,
                       shell=False,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)

    log_file = os.path.join(f_path, 'log.txt')
    with open(log_file, 'w') as f:
        f.write(str(p.stderr))

    check_shotdetection(os.path.join(f_path, 'result.xml'), video_name, p.stderr, p.stdout)


def main(videos_root, features_root, sensitivity, videoids, idmapper):
    # repeat until all movies are processed correctly
    for videoid in tqdm(videoids):
        try:
            video_rel_path = idmapper.get_filename(videoid)
        except KeyError as err:
            raise KeyError("No such videoid: '{videoid}'".format(videoid=err))

        video_name = os.path.basename(video_rel_path)[:-4]
        features_dir = os.path.join(features_root, videoid, EXTRACTOR)
        if not os.path.isdir(features_dir):
            os.makedirs(features_dir)

        done_file_path = os.path.join(features_dir, '.done')

        v_path = os.path.join(videos_root, video_rel_path)

        # create the version for a run, based on the script version and the used parameters
        done_version = VERSION+'\n'+sensitivity

        if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == done_version:
            print('shot detection results missing or version did not match, detecting shots for {video_name}'.format(video_name=video_name))

            # create the folder for the shot-detection, delete the old folder to prevent issues with older versions
            if not os.path.isdir(features_dir):
                os.makedirs(features_dir)
            else:
                shutil.rmtree(features_dir)
                os.makedirs(features_dir)

            shot_detect(v_path, features_dir, sensitivity, video_name)

            # create a hidden file to signal that the asr for a movie is done
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
    parser.add_argument('--sensitivity', default='60', help='sets the sensitivity of the shot_detection, expects an integer ,default value is 60')
    args = parser.parse_args()

    idmapper = TSVIdMapper(args.file_mappings)
    videoids = args.videoids if len(args.videoids) > 0 else parser.error('no videoids found')

    main(args.videos_dir, args.features_dir, args.sensitivity, videoids, idmapper)

