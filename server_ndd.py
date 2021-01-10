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

TRIM_THRESHOLD = 12

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False    # prevent log messages from appearing twice

inception_model = load_model()


def update_index(features_root, index, force_run):

    logger.info('updating feature index')
    # get the path to all features the were extracted correctly
    feature_done_files = glob.glob(os.path.join(features_root, '**', 'features', '.done'), recursive=True)

    for fdf in feature_done_files:
        videoid = os.path.split(os.path.split(os.path.split(fdf)[0])[0])[1]
        done_file_version = open(fdf, 'r').read()

        # get the file_extension of the images from the .done-file
        file_extension = done_file_version.split()[1]

        # check if the features have already been indexed
        if not videoid in index or not index[videoid] == done_file_version or force_run:
            # index features: if there is no entry for the videoid
            # if the version of the indexed features and the new features are different
            # if the force_run flag has been set to true

            # retrieve the path to images
            ip = os.path.join(os.path.split(os.path.split(fdf)[0])[0], 'frames/*{file_extension}'.format(file_extension=file_extension))
            images_path = glob.glob(ip, recursive=True)

            # retrieve the paths to the features and load them with numpy
            fp = os.path.join(os.path.split(fdf)[0], '*.npy')
            features_path = glob.glob(fp, recursive=True)

            video_data = [[np.load(f)[0], images_path[i], videoid, done_file_version] for i, f in enumerate(features_path)]
            # FIXME index is missing the name of the video (can maybe be retrieved from ada.filmontology)
            # FIXME there is no info about from which shot a feature is
            index[videoid] = {'version': done_file_version, 'data': video_data}
        else:
            # if
            # features are already in the index
            pass

    # create a numpy array of the data (features, paths ...) from the index
    # this makes the features sortable, while keeping the videoid, ...
    # delete the data from the to save space
    data = []
    for key in index:
        d = index[key].pop('data', None)
        data = [*data, *d]
    data = np.array(data)

    return index, data


def encode_image_in_base64(image):

    buf = BytesIO()
    image.save(buf, 'PNG')
    encoded_image = buf.getvalue()
    encoded_image = base64.encodebytes(encoded_image).decode('ascii')

    return encoded_image


class RESTHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, index, features_norm_sq, *args, **kwargs):
        logger.debug("RESTHandler::__init__")
        self.index = index
        self.X = index['features']
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

    index = {}
    index, features = update_index(features_root='../static', index=index, force_run=False)

    HOST_NAME = ''
    PORT_NUMBER = 9000

    p = list(features[:, 0])

    print(np.shape(p))

    archive_features_norm_sq = (p**2).sum(axis=1).reshape(-1, 1)

    handler = partial(RESTHandler, index, archive_features_norm_sq)
    server_class = http.server.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), handler)
    logger.info("Starting dummy REST server on %s:%d", HOST_NAME, PORT_NUMBER)
    try:
        logger.info('server ready')
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info('setting up server failed failed')
        pass
