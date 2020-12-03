import subprocess
import argparse
import os
from tqdm import tqdm
import shutil
import xml.etree.ElementTree as ET

VERSION = '20201115'      # the version of the script
EXTRACTOR = 'shotdetection'
STANDALONE = True  # manages the creation of .done-files, if set to false no .done-files are created and the script will always overwrite old results


def check_shotdetection(xml_path, stderr, stdout):
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
        raise Exception('The video is not a valid video. Either the file is corrupted or not a video \n'
                        'Shotdetection error: {stderr} \n'
                        'Shotdetection output: {stdout}'.format(stderr=stderr, stdout=stdout))


def shot_detect(v_path, f_path, sensitivity):
    print('v_path', v_path)
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
    print('stderr: ', p.stderr, '\n')
    print('stdout: ', p.stdout)

    check_shotdetection(os.path.join(f_path, 'result.xml'), p.stderr, p.stdout)


def main(features_root, sensitivity, videoids, force_run):
    # repeat until all movies are processed correctly
    for videoid in tqdm(videoids):

        # the script expects a fixed directory
        video_dir = os.path.join(features_root, videoid, 'media', videoid+'.mp4')
        print('features_root: ', features_root)
        print('videoid: ', videoid)
        print('video_dir: ', video_dir)
        features_dir = os.path.join(features_root, videoid, EXTRACTOR)
        print('features_dir: ', features_dir)
        if not os.path.isdir(features_dir):
            os.makedirs(features_dir)

        # create the version for a run, based on the script version and the used parameters
        done_file_path = os.path.join(features_dir, '.done')
        done_version = VERSION+'\n'+sensitivity

        if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == done_version or force_run:
            print('shot detection results missing or version did not match, detecting shots for video')

            # create the folder for the shot-detection, delete the old folder to prevent issues with older versions
            if not os.path.isdir(features_dir):
                os.makedirs(features_dir)
            else:
                shutil.rmtree(features_dir)
                os.makedirs(features_dir)

            shot_detect(video_dir, features_dir, sensitivity)

            # create a hidden file to signal that the asr for a movie is done
            # write the current version of the script in the file
            if STANDALONE:
                with open(done_file_path, 'w') as d:
                    d.write(done_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument("videoids", help="List of video ids. If empty, entire corpus is iterated.", nargs='*')
    parser.add_argument('--sensitivity', default='60', help='sets the sensitivity of the shot_detection, expects an integer ,default value is 60')
    parser.add_argument("--force_run", default=False, type=bool, help='sets whether the script runs regardless of the version of .done-files')
    args = parser.parse_args()

    main(args.features_dir, args.sensitivity, args.videoids, args.force_run)
