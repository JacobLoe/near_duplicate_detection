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
import pickle
import argparse

TRIM_THRESHOLD = 12

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False    # prevent log messages from appearing twice

parser = argparse.ArgumentParser()
# FIXME: this should be derived from the existing files matching the features found in rder to avoid problems:
parser.add_argument("--file_extension", default='.jpeg', choices=('.jpeg', '.png'), help="use the extension in which the frames were saved, only .png and .jpg are supported, default is .jpg")
args = parser.parse_args()

inception_model = load_model()


def get_features(features_path):

    logger.info('get features')
    source_video = []
    shot_begin_frame = []
    frame_timestamp = []
    frame_path = []

    list_features_path = glob.glob(os.path.join(features_path, '**/*.npy'), recursive=True)

    feature_list = []
    for feature in tqdm(list_features_path, total=len(list_features_path)):
        feature_list.append(np.load(feature)[0])  # create a list of features for the distance calculation

        i_feature = os.path.split(feature)  # get the name and folder of the current feature
        i_segment = os.path.split(i_feature[0])  # get the name and folder of the current segment
        i_movie = os.path.split(os.path.split(i_segment[0])[0])   # get the id and the folder of the movie

        # recreate the path to the image from the path to the feature
        i_frame_path = os.path.join(i_movie[0], i_movie[1], 'frames', i_segment[1], i_feature[1][:-4]+args.file_extension)  # the path to the image corresponding to the feature

        # add information for the frame
        video_rel_path = os.path.join(i_movie[0], i_movie[1], 'media', i_movie[1]+'.mp4')
        video_name = os.path.basename(video_rel_path)[:-4]

        source_video.append(video_name[:8])  # save the first eight chars of the id as the name of the source video
        shot_begin_frame.append(i_segment[1])  # save the beginning timestamp of the shot the feature is from
        frame_timestamp.append(i_feature[1][:-4])  # save the specific timestamp the feature is at
        frame_path.append(i_frame_path)  # save the path of the feature

    features = np.asarray(feature_list)
    info = {'source_video': source_video, 'shot_begin_frame': shot_begin_frame, 'frame_timestamp': frame_timestamp, 'frame_path': frame_path}
    logger.info('done')
    return features, info


def encode_image_in_base64(image):

    buf = BytesIO()
    image.save(buf, 'PNG')
    encoded_image = buf.getvalue()
    encoded_image = base64.encodebytes(encoded_image).decode('ascii')

    return encoded_image


class RESTHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, features, features_norm_sq, *args, **kwargs):
        logger.debug("RESTHandler::__init__")
        self.X = features
        self.X_norm_sq = features_norm_sq
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
        B = np.dot(target_feature,target_feature.T)
        distances = s.X_norm_sq -2*A + B
        logger.info('calculated all distances')
        distances = np.reshape(distances, distances.shape[0])
        logger.debug(distances.shape)

        # sort by distance, ascending
        logger.info("sorting distances")
        indices = np.argsort(distances).tolist()
        lowest_distances = [(distances[i],
                             info_server['source_video'][i],
                             info_server['shot_begin_frame'][i],
                             info_server['frame_timestamp'][i],
                             encode_image_in_base64(Image.open(info_server['frame_bytes'][i]))) for i in indices]
        logger.info('distances are sorted')

        num_results = post_data['num_results']
        filtered_distances = []
        hits = 0    # hits is incremented whenever a distance is added to filtered_distances, if hits is higher than num_results
        shot_hits = set()  # saves the name of the shots that have been added to filtered_distances
        index = 0

        logger.info('filter distances')
        while (hits < num_results) and (index < (len(lowest_distances)-1)):  # repeat filtering until num_results results are found or there are no distances in the list anymore
            # if the current frame and the following frame are from the same video and the same shot, skip the current frame,
            if (lowest_distances[index][2], lowest_distances[index][1]) in shot_hits:
                index += 1
            else:
                shot_hits.add((lowest_distances[index][2], lowest_distances[index][1]))
                filtered_distances.append(lowest_distances[index])
                hits += 1
                index += 1
        logger.info('finished filtering')

        concepts = []
        concepts.extend([
            {
                 'distance': dis.tolist(),
                 'source_video': sv,
                 'shot_begin_frame': str(datetime.timedelta(seconds=int(sbf) / 1000)),
                 'frame_timestamp': str(datetime.timedelta(seconds=int(ft) / 1000)),
                 'frame_path': fp
            }
            for dis, sv, sbf, ft, fp in filtered_distances]
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


if __name__ == '__main__':
    pickle_features = 'features_pickle/features.pickle'
    if os.path.isfile(pickle_features):
        logger.info('load pickled features')
        with open(pickle_features, 'rb') as handle:
            features_server, info_server = pickle.load(handle)
        logger.info('done')
    else:
        features_server, info_server = get_features('static')
        logger.info('save features with pickle')
        if not os.path.isdir('features_pickle'):
            os.makedirs('features_pickle')
        with open(pickle_features, 'wb') as handle:
            pickle.dump([features_server, info_server], handle)
        logger.info('done')

    HOST_NAME = ''
    PORT_NUMBER = 9000
    
    archive_features_norm_sq = (features_server**2).sum(axis=1).reshape(-1,1)

    handler = partial(RESTHandler, features_server, archive_features_norm_sq)
    server_class = http.server.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), handler)
    logger.info("Starting dummy REST server on %s:%d", HOST_NAME, PORT_NUMBER)
    try:
        logger.info('server ready')
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info('setting up server failed failed')
        pass
