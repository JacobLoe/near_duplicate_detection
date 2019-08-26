import http.server
import json
import urllib.parse
import datetime
import logging

from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances
from extract_features import extract_features, load_model

import os
import glob
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
import numpy as np

logger = logging.getLogger(__name__)


def get_features(features_path, target_image):
    model = load_model()

    print('extract feature for target image')
    target_feature = extract_features(model, target_image)
    print('done')

    source_video = []
    shot_begin_frame = []
    frame_timestamp = []
    frame_path = []

    list_features_path = glob.glob(os.path.join(features_path, '**/*.npy'), recursive=True)

    feature_list = []
    for feature in tqdm(list_features_path, total=len(list_features_path)):
        fv = np.load(feature)
        feature_list.append(fv[0])  # create a list of features for the distance calculation

        i_ft = os.path.split(feature)  # the timestamp in ms of the feature
        i_sbf = os.path.split(i_ft[0])  # the timestamp in ms at which the shot starts where the feature is found
        i_sv = os.path.split(os.path.split(i_sbf[0])[0])  # the name of the videofile of the feature

        # recreate the path to the image from the path to the feature
        i_fp = os.path.join(i_sv[0], i_sv[1], 'frames', i_sbf[1], i_ft[1][:-4]+'.png')  # the path to image corresponding to the feature

        # add information for the frame
        source_video.append(i_sv[1])  # save the name of the source video
        shot_begin_frame.append(i_sbf[1])  # save the beginning timestamp of the shot the feature is from
        frame_timestamp.append(i_ft[1][:-4])  # save the specific timestamp the feature is at
        frame_path.append(i_fp)  # save the path of the feature

    features = {'feature_list': feature_list, 'target_feature': target_feature}
    info = {'source_video': source_video, 'shot_begin_frame': shot_begin_frame, 'frame_timestamp': frame_timestamp, 'frame_path': frame_path}
    return features, info


class RESTHandler(http.server.BaseHTTPRequestHandler):
    def do_HEAD(s):
        print('do_HEAD')
        s.send_response(200)
        s.send_header("Content-type", "application/json")
        s.end_headers()

    def do_GET(s):
        print('do_GET')
        s.send_response(200)
        s.send_header("Content-type", "application/json")
        s.end_headers()

        response = json.dumps({"status": 200, "message": "OK", "data": { ## FIXME alles nach ok kann weg,
            "capabilities": {
                "minimum_batch_size": 1,  # # of frames
                "maximum_batch_size": 500,  # # of frames
                "available_models": 'models'
            }
        }})
        s.wfile.write(response.encode())

    def do_POST(s):
        print('do_POST')
        length = int(s.headers['Content-Length'])
        body = s.rfile.read(length).decode('utf-8')
        if s.headers['Content-type'] == 'application/json':
            post_data = json.loads(body)			# enthält alle daten, target_image usw. die der server braucht
        else:
            post_data = urllib.parse.parse_qs(body)

        # CACHE_DIR = "/var/ndd/cache"
        # # create cachedir
        # if not os.path.exists(CACHE_DIR):
        #     os.makedirs(CACHE_DIR)

        print('load target_image')
        image_scale = 299
        target_image = post_data['target_image']
        target_image = Image.open(BytesIO(base64.b64decode(target_image)))
        target_image = target_image.resize((image_scale, image_scale))
        target_image.save('target_image.png')
        target_image = np.array(target_image)
        print('done')

        features_path = 'data'
        features, info = get_features(features_path, target_image)

        # calculate the distance for all features
        print('calculating the distance for all features')
        distances = pairwise_distances(features['feature_list'], features['target_feature'], metric=euclidean, n_jobs=post_data['num_cores'])
        print('done')
        # sort by distance, ascending
        lowest_distances = sorted(zip(distances, info['source_video'], info['shot_begin_frame'], info['frame_timestamp'], info['frame_path']))

        num_results = post_data['num_results']
        filtered_distances = []
        hits = 0    # hits is incremented whenever a distance is added to filtered_distances, if hits is higher than num_results
        shot_hits = []  # saves the name of the shots that have been added to filtered_distances
        index = 0
        while (hits < num_results) and (index < (len(lowest_distances)-2)):  # repeat filtering until num_results results are found or there are no distances in the list anymore
            # if the current frame and the following frame are from the same video and the same shot, skip the current frame,
            # otherwise add the distance to the list, increment the
            if (lowest_distances[index][1] == lowest_distances[index + 1][1]) and (lowest_distances[index][2] in shot_hits):
                # print(index)
                index += 1
            else:
                shot_hits.append(lowest_distances[index][2])
                filtered_distances.append(lowest_distances[index])
                hits += 1
                index += 1

        concepts = []
        concepts.extend([
            {
                 'distance': dis.tolist(),
                 'source_video': sv,
                 'shot_begin_frame': sbf,
                 'frame_timestamp': str(datetime.timedelta(seconds=int(ft) / 1000)),
                 'frame_path': str(datetime.timedelta(seconds=int(sbf) / 1000))
            }
            for dis, sv, sbf, ft, fp in filtered_distances]
        )

        logger.debug(concepts)
        s.send_response(200)
        s.send_header("Content-type", "application/json")
        s.end_headers()
        response = json.dumps({
            "status": 200,
            "message": "OK",
            "data": concepts
        })
        s.wfile.write(response.encode())


if __name__ == '__main__':

    HOST_NAME = ''
    PORT_NUMBER = 9000

    logging.basicConfig(level=logging.DEBUG)
    server_class = http.server.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), RESTHandler)
    logger.info("Starting dummy REST server on %s:%d", HOST_NAME, PORT_NUMBER)
    try:
        print('server ready')
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('setting up server failed failed')
        pass
    # httpd.server_close()
