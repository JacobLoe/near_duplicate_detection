import logging

logger = logging.getLogger(__name__)

import http.server
import json
import urllib.parse
import os

from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances

import datetime

import numpy as np
from PIL import Image
from io import BytesIO
import base64
import itertools

from query import get_features
CACHE_DIR = "/var/vcd/cache"

# models = [
#     {
#         "id": "standard",  # FIXME: improve selection of default model
#         "label": "ResNet50",
#         "image_size": 224
#     },
#     {
#         "id": "resnet50",
#         "label": "ResNet50",
#         "image_size": 224
#     },
# ]
#
# model_impls = {
#     "standard": {  # FIXME: s.a.
#         "class": ResNet50,
#         "params": {'weights': 'imagenet', },
#     },
#     "resnet50": {
#         "class": ResNet50,
#         "params": {'weights': 'imagenet', },
#     }
# }

top_n_preds = 3

# create cachedir
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


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
            post_data = json.loads(body)			# enthät alle daten, target_image usw. die der server braucht
        else:
            post_data = urllib.parse.parse_qs(body)

        target_size = 299 #(dict([(m["id"], m['image_size']) for m in models]))[modelid]
        concepts = []

        print('post_data', post_data)
        features, info = get_features(post_data['features_path'], post_data['target_image'])

# loop kann (vermutlich) weg
#         for annotation in post_data['annotations']:
#             print(annotation)
#             aid = annotation['annotationid']
#             begin = annotation['begin']
#             begin = annotation['end']
#
# # batch_x ist das target_image
#             batch_x = np.zeros((1, target_size, target_size, 3), dtype=np.float32)
#
# # die images in batch_x werden predicted
# # loop kann (vermutlich) weg
#             for i, frame in enumerate(annotation['frames']):
#                 # Load image to PIL format
#                 img = Image.open(BytesIO(base64.b64decode(frame['screenshot'])))
#                 # cache frame - FIXME: currently there is no mean to identify the video - same timstamp will overwrite an old frame (hash?)
#                 img.save(os.path.join(CACHE_DIR, '{0}.png'.format(frame['timecode'])))
#                 if img.mode != 'RGB':
#                     img = img.convert('RGB')
#                 hw_tuple = (target_size, target_size)
#                 if img.size != hw_tuple:
#                     logger.warn("Scaling image to model size - this should be done in advene!")
#                     img = img.resize(hw_tuple)
#                 x = image.img_to_array(img)
#                 x = np.expand_dims(x, axis=0)
#                 x = preprocess_input(x)
#                 batch_x[i] = x[0, :, :, :]
#             #preds = model.predict_on_batch(np.asarray(batch_x))
#
## ergebnisse concepts liste als dictionary, jede zeile in der html-tabelle ist ein element (als dict) in der list
        num_cores = 4
        # calculate the distance for all features
        d_distances = pairwise_distances(features['feature_list'], features['target_feature'], metric=euclidean, n_jobs=num_cores)
        # sort by distance, ascending
        lowest_distances = sorted(zip(d_distances, info['source_video'], info['shot_begin_frame'], info['frame_timestamp'], info['frame_path']))

        num_results = 30
        filtered_distances = lowest_distances[:num_results]

        print(len(filtered_distances))

        concepts.extend({
            'results': [
                {'distance': dis,
                 'source_video': sv,
                 'shot_begin_frame': sbf,
                 'frame_timestamp': str(datetime.timedelta(seconds=int(ft) / 1000)),
                 'frame_path': str(datetime.timedelta(seconds=int(sbf) / 1000))
                }
                for dis, sv, sbf, ft, fp in filtered_distances]
            }
        )
            # concepts.extend({
            #     'target_feature': features['target_feature'],
            #     'annotations': [
            #         {'feature': feature,
            #             'source_video': info['source_video'][i],
            #             'shot_begin_frame': info['shot_begin_frame'][i],
            #             'frame_timestamp': info['frame_timestamp'][i],
            #             'frame_path': info['frame_path'][i]
            #         }
            #         for i, feature in enumerate(features['feature_list'])]
            #     }
            # )
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
