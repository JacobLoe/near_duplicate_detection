import http.server
import json
import urllib.parse
import datetime
import logging

from extract_features import extract_features, load_model
from crop_image import trim

from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing as mp
import functools

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
logging.basicConfig(level=logging.DEBUG)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False    # prevent log messages from appearing twice

parser = argparse.ArgumentParser()
parser.add_argument("--num_cores", type=int, default=4, help="specify the number cpu cores used for distance calculation, default value is 4")
parser.add_argument("--features_dtype", default='float16', help="set the dtype of the features, default is float16")
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
        fv = np.load(feature)
        feature_list.append(fv[0])  # create a list of features for the distance calculation

        i_ft = os.path.split(feature)  # the timestamp in ms of the feature
        i_sbf = os.path.split(i_ft[0])  # the timestamp in ms at which the shot starts where the feature is found
        i_sv = os.path.split(os.path.split(i_sbf[0])[0])  # the name of the videofile of the feature

        # recreate the path to the image from the path to the feature
        i_fp = os.path.join(i_sv[0], i_sv[1], 'frames', i_sbf[1], i_ft[1][:-4]+'.jpg')  # the path to image corresponding to the feature

        # add information for the frame
        source_video.append(i_sv[1])  # save the name of the source video
        shot_begin_frame.append(i_sbf[1])  # save the beginning timestamp of the shot the feature is from
        frame_timestamp.append(i_ft[1][:-4])  # save the specific timestamp the feature is at
        frame_path.append(i_fp)  # save the path of the feature

    features = {'feature_list': feature_list}
    info = {'source_video': source_video, 'shot_begin_frame': shot_begin_frame, 'frame_timestamp': frame_timestamp, 'frame_path': frame_path}
    logger.info('done')
    return features, info


pickle_features = 'features_pickle/features.pickle'
if os.path.isfile(pickle_features):
    logger.info('load pickled features')
    with open(pickle_features, 'rb') as handle:
        features_server, info_server = pickle.load(handle)
    logger.info('done')
else:
    features_server, info_server = get_features('static')
    logger.info('save features with pickle')
    with open(pickle_features, 'wb') as handle:
        pickle.dump([features_server, info_server], handle)
    logger.info('done')

archive_features = np.asarray(features_server['feature_list'], dtype=args.features_dtype)


def compute_batch(start_idx, batch_size, Y):

    global archive_features

    dists = euclidean_distances(archive_features[start_idx:start_idx+batch_size, :], Y)
    return dists


class RESTHandler(http.server.BaseHTTPRequestHandler):

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

        buf = BytesIO()
        target_image.save(buf, 'PNG')
        trimmed_target_image = buf.getvalue()
        trimmed_target_image = base64.encodebytes(trimmed_target_image).decode('ascii')

        target_image = target_image.resize((image_scale, image_scale))
        target_image.save('target_image.png')
        target_image = np.array(target_image)
        logger.info('finished loading target image')

        logger.info('extract feature for target image')
        target_feature = extract_features(inception_model, target_image)
        logger.info('done')

        # calculate the distance for all features
        logger.info('calculating the distance for all features')
        logger.debug(archive_features.shape)

        f_size = archive_features.shape[0]   # the amount of features
        nproc = args.num_cores if args.num_cores > 0 else mp.cpu_count()    # number of cpu jobs
        logger.info("Using {nproc} cpus".format(nproc=nproc))
        batch_size = int(float(f_size)/nproc)

        compute_batch_ = functools.partial(compute_batch, batch_size=batch_size, Y=target_feature)

        with mp.Pool(nproc) as pool:
            distances = list(pool.imap(compute_batch_, [i for i in range(0, f_size, batch_size)]))

        distances = np.concatenate(distances)
        distances = np.reshape(distances, distances.shape[0])
        logger.info('calculated all distances')

        # sort by distance, ascending
        logger.info("sorting distances")
        indices = np.argsort(distances).tolist()
        lowest_distances = [(distances[i], info_server['source_video'][i], info_server['shot_begin_frame'][i], info_server['frame_timestamp'][i], info_server['frame_path'][i]) for i in indices]
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

    HOST_NAME = ''
    PORT_NUMBER = 9000

    server_class = http.server.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), RESTHandler)
    logger.info("Starting dummy REST server on %s:%d", HOST_NAME, PORT_NUMBER)
    try:
        print('server ready')
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('setting up server failed failed')
        pass
