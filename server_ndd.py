import http.server
import json
import urllib.parse
import datetime
import logging
from functools import partial

from extract_features import extract_features, load_model
from crop_image import trim

import os
import glob
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import requests


TRIM_THRESHOLD = 12

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False    # prevent log messages from appearing twice

inception_model = load_model()


def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root = tree.getroot().findall('content')
    timestamps = []
    for child in root[0].iter():
        if child.tag == 'shot':
            attribs = child.attrib

            timestamps.append((int(attribs['msbegin']), int(attribs['msbegin']) + int(attribs['msduration']) - 1))
    return timestamps


def update_index(features_root, video_index, features, video_data, force_run):
    logger.info('updating feature index')
    # get the path to all features the were extracted correctly
    feature_done_files = glob.glob(os.path.join(features_root, '**', 'features', '.done'), recursive=True)

    for fdf in tqdm(feature_done_files):
        # as the videoid is taken from the feature_done_file path
        videoid = os.path.split(os.path.split(os.path.split(fdf)[0])[0])[1]
        done_file_version = open(fdf, 'r').read()

        # get the file_extension of the images from the .done-file
        file_extension = done_file_version.split()[1]

        # check if the features have already been indexed
        if not videoid in video_index or not video_index[videoid]['version'] == done_file_version or force_run == 'True':
            # index features: if there is no entry for the videoid
            # if the version of the indexed features and the new features are different
            # if the force_run flag has been set to true

            # get the name of the video from ada the server
            try:
                media_base_url = "http://ada.filmontology.org/api_dev/media/"
                r = requests.get(media_base_url + videoid)
                if r.status_code == 200:
                    data = r.json()
                    video_url = data.get('videourl')
                    videoname = os.path.split(video_url)[1][:-4]
            except:
                videoname = videoid

            # retrieve the path to images
            ip = os.path.join(os.path.split(os.path.split(fdf)[0])[0], 'frames/*{file_extension}'.format(file_extension=file_extension))
            images_path = glob.glob(ip, recursive=True)

            # retrieve the paths to the features and load them with numpy
            fp = os.path.join(os.path.split(fdf)[0], '*.npy')
            features_path = glob.glob(fp, recursive=True)

            # read the shotdetect results to map frames to shots
            shotdetect_path = os.path.join(features_root, videoid, 'shotdetect/result.xml')
            shot_timestamps = read_shotdetect_xml(shotdetect_path)

            aux_features = []
            aux_video_data = []
            for i, f in enumerate(features_path):
                aux_features.append(np.load(f)[0])
                frame_timestamp = os.path.split(images_path[i])[1][:-len(file_extension)]

                for ts in shot_timestamps:
                    if ts[0] < int(frame_timestamp) < ts[1]:
                        shot_begin_timestamp = ts[0]

                aux_video_data.append({'image_path': images_path[i], 'frame_timestamp': frame_timestamp,
                                       'shot_begin_timestamp': shot_begin_timestamp, 'videoname': videoname,
                                       'videoid': videoid, 'version': done_file_version})

            video_index[videoid] = {'version': done_file_version, 'features': aux_features, 'video_data': aux_video_data}
        else:
            # if features were already indexed return them from the "features" list to the index dict

            # in the video_data search for the data and corresponding index for the current videoid
            ivd = [[i, vd] for i, vd in enumerate(video_data) if vd['videoid'] == videoid]

            # fill the video_index in the same way as in the upper if condition
            video_index[videoid]['video_data'] = []
            video_index[videoid]['features'] = []
            for i, vd in ivd:
                video_index[videoid]['video_data'].append(vd)
                video_index[videoid]['features'].append(features[i])

    # create a numpy array of the data (features, paths ...) from the index
    # this makes the features sortable, while keeping the videoid, ...
    # delete the data (except the version) from the index to save space
    features = []
    video_data = []
    videos_with_no_features = []
    for key in video_index:
        # if features were removed in between two updates (meaning they are in the "features"  list but the features don't exist on the disk)
        # remember the key and remove them later from dict
        if not 'features' in video_index[key]:
            videos_with_no_features.append(key)
        else:
            f = video_index[key].pop('features', None)
            d = video_index[key].pop('video_data', None)
            features = [*features, *f]
            video_data = [*video_data, *d]
    features = np.asarray(features)

    # remove unneeded keys
    for key in videos_with_no_features:
        del video_index[key]

    return video_index, features, video_data

def encode_image_in_base64(image):

    buf = BytesIO()
    image.save(buf, 'PNG')
    encoded_image = buf.getvalue()
    encoded_image = base64.encodebytes(encoded_image).decode('ascii')

    return encoded_image


class RESTHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, video_index, features, video_data, features_norm_sq, features_root, *args, **kwargs):
        logger.debug("RESTHandler::__init__")
        self.video_index = video_index
        self.video_data = video_data
        self.X = features
        self.X_norm_sq = features_norm_sq
        self.features_root = features_root
        super().__init__(*args, **kwargs)

    def do_HEAD(s):
        s.send_response(200)
        s.send_header("Content-type", "application/json")
        s.end_headers()

    def do_GET(s):
        s.send_response(200)
        s.send_header("Content-type", "application/json")
        s.end_headers()

        response = json.dumps({"status": 200, "message": "OK"})
        s.wfile.write(response.encode())

    def do_POST(s):
        length = int(s.headers['Content-Length'])
        body = s.rfile.read(length).decode('utf-8')
        if s.headers['Content-type'] == 'application/json':
            post_data = json.loads(body)
        else:
            post_data = urllib.parse.parse_qs(body)

        if not post_data['update_index']:
            # calculate the nearest neighbours the target image
            logger.info('load target_image')
            image_scale = 299
            target_image = post_data['target_image']
            target_image = Image.open(BytesIO(base64.b64decode(target_image)))
            if post_data['remove_letterbox'] == 'True':
                logger.info('removed letterbox in target image')
                target_image, _ = trim(target_image, TRIM_THRESHOLD)

            trimmed_target_image = encode_image_in_base64(target_image)

            target_image = target_image.resize((image_scale, image_scale))
            target_image.save('target_image.png')
            target_image = np.array(target_image)
            logger.info('finished loading target image')

            logger.info('extract feature for target image')
            target_feature = extract_features(inception_model, target_image)
            logger.info('done')

            # calculate the distance for all features
            logger.info('calculating the distance for all features')
            logger.debug(s.X.shape)

            A = np.dot(s.X, target_feature.T)
            B = np.dot(target_feature, target_feature.T)
            distances = s.X_norm_sq -2*A + B
            logger.info('calculated all distances')
            distances = np.reshape(distances, distances.shape[0])
            logger.debug(distances.shape)

            # sort by distance, ascending
            logger.info("sorting distances")
            indices = np.argsort(distances).tolist()    # returns the indices of the sorted distances
            lowest_distances = [(distances[i],
                                 s.video_data[i]['videoid'],
                                 s.video_data[i]['shot_begin_timestamp'],
                                 s.video_data[i]['frame_timestamp'],
                                 encode_image_in_base64(Image.open(s.video_data[i]['image_path']))) for i in indices]
            logger.info('distances are sorted')

            # filter the distance such that only num_results are shown and each shot in a video appears only once
            num_results = post_data['num_results']
            filtered_distances = []
            hits = 0    # hits is incremented whenever a distance is added to filtered_distances, if hits is higher than num_results
            shot_hits = set()  # saves the shot timestamp and the name of the video for the first num_results distances
            i = 0

            logger.info('filter distances')
            while (hits < num_results) and (i < (len(lowest_distances)-1)):  # repeat filtering until num_results results are found or there are no distances in the list anymore
                # if the current frame and the following frame are from the same video and the same shot, skip the current frame,
                if (lowest_distances[i][2], lowest_distances[i][1]) in shot_hits:
                    i += 1
                else:
                    shot_hits.add((lowest_distances[i][2], lowest_distances[i][1]))
                    filtered_distances.append(lowest_distances[i])
                    hits += 1
                    i += 1
            logger.info('finished filtering')

            concepts = []
            concepts.extend([
                {
                    'distance': dis.tolist(),
                    'source_video': sv,
                    'frame_timestamp': str(datetime.timedelta(seconds=int(ft) / 1000)),
                    'shot_begin_timestamp': str(datetime.timedelta(seconds=int(sbt) / 1000)),
                    'frame_bytes': fb
                }
                for dis, sv, ft, sbt, fb in filtered_distances]
            )

            s.send_response(200)
            s.send_header("Content-type", "application/json")
            s.end_headers()
            response = json.dumps({
                "status": 200,
                "message": "OK",
                "data": concepts,
                "trimmed_target_image": trimmed_target_image
            })
            s.wfile.write(response.encode())
        else:
            # update the index and the features
            video_index, features, video_data = update_index(features_root=s.features_root,
                                                             video_index=s.video_index,
                                                             features=s.X,
                                                             video_data=s.video_data,
                                                             force_run=post_data['force_run'])

            s.video_index = video_index
            s.video_data = video_data
            s.X = features
            s.X_norm_sq = (features ** 2).sum(axis=1).reshape(-1, 1)

            s.send_response(200)
            s.send_header("Content-type", "application/json")
            s.end_headers()
            response = json.dumps({
                "status": 200,
                "message": "OK"
            })
            s.wfile.write(response.encode())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("features_root", help="the directory where the images are to be stored")
    args = parser.parse_args()

    video_index = {}
    features = []
    video_data = []
    video_index, features, video_data = update_index(features_root=args.features_root,
                                                     video_index=video_index,
                                                     features=features,
                                                     video_data=video_data,
                                                     force_run=False)

    HOST_NAME = ''
    PORT_NUMBER = 9000

    archive_features_norm_sq = (features**2).sum(axis=1).reshape(-1, 1)

    handler = partial(RESTHandler, video_index, features, video_data, archive_features_norm_sq, args.features_root)
    server_class = http.server.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), handler)
    logger.info("Starting dummy REST server on %s:%d", HOST_NAME, PORT_NUMBER)
    try:
        logger.info('server ready')
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info('setting up server failed failed')
        pass
