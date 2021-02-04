import subprocess
import argparse
import os
from tqdm import tqdm
import csv
import shutil
import xml.etree.ElementTree as ET
import logging

VERSION = '20210204'      # the date the script was last changed
EXTRACTOR = 'shotdetect'
STANDALONE = True  # manages the creation of .done-files, if set to false no .done-files are created and the script will always overwrite old results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False    # prevent log messages from appearing twice


def extractShots(xmlfile, csvfile):
    tree = ET.parse(xmlfile)
    content = tree.getroot().find('content')
    media = content.find('head').find('media')
    shots = content.find('body').find('shots')
    
    logger.debug('Write shotdetect csv file for {xmlfile}'.format(xmlfile=xmlfile))
    
    with open(csvfile, "wt") as of:
        tsv_writer = csv.writer(of, delimiter='\t')
        for shot in shots:
            attribs = shot.attrib
            begin = int(attribs.get('msbegin'));
            end = begin + int(attribs.get('msduration'))
            content = attribs.get('id')
            tsv_writer.writerow([begin, end, content]);


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


def shotdetect(v_path, f_path, sensitivity):
    logger.debug('v_path', v_path)
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
    logger.debug('stderr: ', p.stderr, '\n')
    logger.debug('stdout: ', p.stdout)

    check_shotdetection(os.path.join(f_path, 'result.xml'), p.stderr, p.stdout)


def main(features_root, sensitivity, videoids, force_run):
    # repeat until all movies are processed correctly
    for videoid in tqdm(videoids):

        # the script expects a fixed directory
        video_dir = os.path.join(features_root, videoid, 'media', videoid+'.mp4')
        logger.debug('features_root: ', features_root)
        logger.debug('videoid: ', videoid)
        logger.debug('video_dir: ', video_dir)
        features_dir = os.path.join(features_root, videoid, EXTRACTOR)
        logger.debug('features_dir: ', features_dir)
        if not os.path.isdir(features_dir):
            os.makedirs(features_dir)

        # create the version for a run, based on the script version and the used parameters
        done_file_path = os.path.join(features_dir, '.done')
        done_version = VERSION+'\n'+sensitivity

        if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == done_version or force_run == 'True':
            logger.info('shot detection results missing or version did not match, detecting shots for video')

            # create the folder for the shot-detection, delete the old folder to prevent issues with older versions
            if not os.path.isdir(features_dir):
                os.makedirs(features_dir)
            else:
                shutil.rmtree(features_dir)
                os.makedirs(features_dir)

            shotdetect(video_dir, features_dir, sensitivity)

            # convert results.xml to csv
            xmlfile = os.path.join(features_dir, 'result.xml')
            csvfile = os.path.join(features_dir, '{vid}.csv'.format(vid=videoid))
            extractShots(xmlfile,csvfile)

            # create a hidden file to signal that the asr for a movie is done
            # write the current version of the script in the file
            if STANDALONE:
                with open(done_file_path, 'w') as d:
                    d.write(done_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_root", help="the directory where the images are to be stored")
    parser.add_argument("videoids", help="List of video ids. If empty, entire corpus is iterated.", nargs='*')
    parser.add_argument('--sensitivity', default='60', help='sets the sensitivity of the shot_detection, expects an integer ,default value is 60')
    parser.add_argument("--force_run", default='False', help='sets whether the script runs regardless of the version of .done-files')
    args = parser.parse_args()

    main(args.features_root, args.sensitivity, args.videoids, args.force_run)
