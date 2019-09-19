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
inception_model = load_model()


def get_features(features_path):

    print('get features')
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

    features = {'feature_list': feature_list}
    info = {'source_video': source_video, 'shot_begin_frame': shot_begin_frame, 'frame_timestamp': frame_timestamp, 'frame_path': frame_path}
    print('done')
    return features, info


features_server, info_server = get_features('static')


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

        response = json.dumps({"status": 200, "message": "OK"})
        s.wfile.write(response.encode())

    def do_POST(s):
        print('do_POST')
        length = int(s.headers['Content-Length'])
        body = s.rfile.read(length).decode('utf-8')
        if s.headers['Content-type'] == 'application/json':
            post_data = json.loads(body)
        else:
            post_data = urllib.parse.parse_qs(body)

        # CACHE_DIR = "/var/ndd/cache"
        # # create cachedir
        # if not os.path.exists(CACHE_DIR):
        #     os.makedirs(CACHE_DIR)

        logger.info('load target_image')
        image_scale = 299
        target_image = post_data['target_image']
        target_image = Image.open(BytesIO(base64.b64decode(target_image)))
        target_image = target_image.resize((image_scale, image_scale))
        target_image.save('target_image.png')
        target_image = np.array(target_image)
        logger.info('finished loading target image')

        logger.info('extract feature for target image')
        target_feature = extract_features(inception_model, target_image)
        logger.info('done')

        # calculate the distance for all features
        logger.info('calculating the distance for all features')
        num_cores = 8
        distances = pairwise_distances(features_server['feature_list'], target_feature, metric=euclidean, n_jobs=num_cores)
        logger.info('calculated all distances')
        # sort by distance, ascending
        lowest_distances = sorted(zip(distances, info_server['source_video'], info_server['shot_begin_frame'], info_server['frame_timestamp'], info_server['frame_path']))

        num_results = post_data['num_results']
        filtered_distances = []
        hits = 0    # hits is incremented whenever a distance is added to filtered_distances, if hits is higher than num_results
        shot_hits = set()  # saves the name of the shots that have been added to filtered_distances
        index = 0
        aa = []

        while (hits < num_results) and (index < (len(lowest_distances)-1)):  # repeat filtering until num_results results are found or there are no distances in the list anymore
            # if the current frame and the following frame are from the same video and the same shot, skip the current frame,
            if (lowest_distances[index][2], lowest_distances[index[1]]) in shot_hits:
                index += 1
            else:
                aa.append((lowest_distances[index][2], lowest_distances[index[1]]) in shot_hits)
                shot_hits.add(lowest_distances[index][2], lowest_distances[index[1]])
                filtered_distances.append(lowest_distances[index])
                hits += 1
                index += 1
        # print(aa)
        # print(shot_hits)
        # print('len_shot_hits: ', len(shot_hits))
        # print('hits: ', hits)
        # print('len_dists: ', len(lowest_distances[0]))
        # print('index: ', index)

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

        # logger.debug(concepts)
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
