import json
import datetime
import logging

from extract_features import extract_features, load_model
from utils import trim, read_shotdetect

import os
import glob
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import requests

from flask import Flask, request

from waitress import serve

TRIM_THRESHOLD = 12  # the threshold for the trim function, pixels with values lower are considered black and croppped
TARGET_WIDTH = 480  # all images that are sent to the client will be scaled to this width (while keeping the aspect ratio intact)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False  # prevent log messages from appearing twice

app = Flask(__name__)
app.config['SECRET_KEY'] = b'\x8bJS@\x8e\xd2\x17Q:\xe9w\x13\r\x8d\xc7\xae'


def resize_image(image_path, remove_letterbox=False, target_width=TARGET_WIDTH):
    """
    Resizes an image to a target width according to its aspect ratio, if needed removes the black border

    :param image_path: type string, the path to the image
    :param remove_letterbox: type bool, default False, removes black border around the image
    :param target_width: type int, default is TARGET_WIDTH specified at the top of the script, defines the width to which the image is scaled
    :return:    PIL image object
    """
    image = Image.open(image_path)

    if remove_letterbox == 'True':
        image, _ = trim(image, TRIM_THRESHOLD)

    # downsize the images if their width is bigger than the target_width
    # to make the response of the server smaller
    if target_width:
        resolution = np.shape(image)
        ratio = resolution[1] / resolution[0]
        frame_height = int(TARGET_WIDTH / ratio)
        resolution_new = (TARGET_WIDTH, frame_height)
        image = image.resize(resolution_new)

    return image


def encode_image_in_base64(image):
    """
    Takes an image opened with PIL and encodes it into base64
    :param image:   PIL image object
    :return:    a bytes object as string
    """
    buf = BytesIO()
    image.save(buf, 'JPEG')
    encoded_image = buf.getvalue()
    encoded_image = base64.encodebytes(encoded_image).decode('ascii')

    return encoded_image


class NearDuplicateDetection:
    def __init__(self, features_root):
        """
        Initializes the variables of a NearDuplicateDetection object. Loads the model for ..
        Creates the index from scratch (either with empty variables or from the already existing files)
        :param features_root: type string, defines the directory under which all files (video, images, shotdetection, features) for the ndd are saved
        """
        self.video_index = {}
        self.video_data = []
        self.features = []
        self.features_root = features_root
        self.features_norm_sq = []

        # load the model for computing features from images
        logger.info('loading inception model')
        self.inception_model = load_model()

        # update the index for the first time
        self.update_index()

    def calculate_distance(self, target_feature, num_results):
        """
        Computes and returns the nearest neighbours to the query feature
        :param target_feature: type numpy array, the feature array of the query image
        :param num_results: type int, defines how many results are to be returned
        :return:
        """
        # calculate the distance for all features
        logger.info('calculating the distance for all features')
        logger.debug(self.features.shape)

        A = np.dot(self.features, target_feature.T)
        B = np.dot(target_feature, target_feature.T)
        distances = self.features_norm_sq - 2 * A + B
        logger.info('calculated all distances')
        distances = np.reshape(distances, distances.shape[0])
        logger.debug(distances.shape)

        # sort by distance, ascending
        logger.info("sorting distances")
        indices = np.argsort(distances).tolist()  # returns the indices of the sorted distances
        lowest_distances = [(distances[i],
                             self.video_data[i]['videoid'],
                             self.video_data[i]['shot_begin_timestamp'],
                             self.video_data[i]['frame_timestamp'],
                             resize_image(self.video_data[i]['image_path']))
                            for i in indices]
        logger.info('distances are sorted')

        # filter the distance such that only num_results are shown and each shot in a video appears only once
        filtered_distances = []
        hits = 0  # hits is incremented whenever a distance is added to filtered_distances, if hits is higher than num_results
        shot_hits = set()  # saves the shot timestamp and the name of the video for the first num_results distances
        i = 0

        logger.info('filter distances')
        while (hits < num_results) and (i < (len(lowest_distances) - 1)):  # repeat filtering until num_results results are found or there are no distances in the list anymore
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
                'frame': f
            }
            for dis, sv, ft, sbt, f in filtered_distances]
        )

        return concepts

    def update_index(self, videoids=[]):
        """

        :param videoids:
        """
        logger.info('updating feature index')

        # FIXME
        # if no videoids were supplied find all features that were already extracted correctly (meaning a .done-file exists)
        if len(videoids) == 0:
            feature_done_files = glob.glob(os.path.join(self.features_root, '**', 'features', '.done'), recursive=True)
            videoids = [os.path.split(os.path.split(os.path.split(fdf)[0])[0])[1] for fdf in feature_done_files]

        if len(videoids) != 0:
            for videoid in tqdm(videoids):

                features_dir = os.path.join(self.features_root, videoid)

                done_file_version = open(os.path.join(features_dir, 'features', '.done'), 'r').read()

                # get the file_extension of the images from the .done-file
                file_extension = done_file_version.split()[1]

                # check if the features have already been indexed
                if not videoid in self.video_index or not self.video_index[videoid]['version'] == done_file_version:
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
                    ip = os.path.join(features_dir, 'frames/*.{file_extension}'.format(file_extension=file_extension))
                    images_path = glob.glob(ip, recursive=True)

                    # retrieve the paths to the features and load them with numpy
                    fp = os.path.join(features_dir, 'features', '*.npy')
                    features_path = glob.glob(fp, recursive=True)

                    # read the shotdetect results to map frames to shots
                    shotdetect_file_path = os.path.join(self.features_root, '{videoid}/shotdetect/{videoid}.csv'.format(videoid=videoid))
                    shot_timestamps = read_shotdetect(shotdetect_file_path)

                    # FIXME add comment
                    aux_features = []
                    aux_video_data = []
                    for i, f in enumerate(features_path):
                        aux_features.append(np.load(f)[0])
                        frame_timestamp = os.path.splitext(os.path.split(images_path[i])[1])[0]

                        for ts in shot_timestamps:
                            #
                            if ts[0] < int(frame_timestamp) < ts[1]:
                                shot_begin_timestamp = ts[0]

                        aux_video_data.append({'image_path': images_path[i], 'frame_timestamp': frame_timestamp,
                                               'shot_begin_timestamp': shot_begin_timestamp, 'videoname': videoname,
                                               'videoid': videoid, 'version': done_file_version})

                    self.video_index[videoid] = {'version': done_file_version, 'features': aux_features, 'video_data': aux_video_data}
                else:
                    # FIXME clarify
                    # if features were already indexed return them from the "features" list to the index dict

                    # in the video_data search for the data and corresponding index for the current videoid
                    ivd = [[i, vd] for i, vd in enumerate(self.video_data) if vd['videoid'] == videoid]

                    # fill the video_index in the same way as in the above if condition
                    self.video_index[videoid]['video_data'] = []
                    self.video_index[videoid]['features'] = []
                    for i, vd in ivd:
                        self.video_index[videoid]['video_data'].append(vd)
                        self.video_index[videoid]['features'].append(self.features[i])

            # create a numpy array of the data (features, paths ...) from the index
            # this makes the features sortable, while keeping the videoid, ... FIXME
            # delete the data (except the version) from the index to save space
            features = []
            video_data = []
            videos_with_no_features = []
            for key in self.video_index:
                if not 'features' in self.video_index[key]:
                    # if features were removed in between two updates (meaning they are in the "features" list but the features don't exist on the disk)
                    # remember the key and remove them later from the dict
                    videos_with_no_features.append(key)
                else:
                    f = self.video_index[key].pop('features', None)
                    d = self.video_index[key].pop('video_data', None)
                    features = [*features, *f]
                    video_data = [*video_data, *d]
            features = np.asarray(features)

            # remove unneeded keys
            for key in videos_with_no_features:
                del self.video_index[key]

            self.video_data = video_data
            self.features = features
            self.features_norm_sq = (self.features ** 2).sum(axis=1).reshape(-1, 1)

        else:
            logger.info('No features were extracted yet. Server is not ready to calculate nearest neighbours until the index is updated with features')


@app.route('/', methods=['POST'])
def process_file():
    post_data = request.json

    if not post_data['update_index']:
        # calculate the nearest neighbours the target image
        logger.info('load query image')
        query_image = post_data['query_image']
        query_image = BytesIO(base64.b64decode(query_image))

        # FIXME maybe suppress the resizing, to prevent double resize for feature extraction, suppression may lead to trouble with displaying on the client
        query_image = resize_image(query_image, remove_letterbox=post_data['remove_letterbox'])

        resized_query_image = query_image.resize((299, 299))
        resized_query_image = np.array(resized_query_image)
        logger.info('finished loading query image')

        logger.info('extract feature for query image')
        query_feature = extract_features(ndd.inception_model, resized_query_image)

        concepts = ndd.calculate_distance(target_feature=query_feature, num_results=post_data['num_results'])

        # encode query image to send it back to the client
        query_image_bytes = encode_image_in_base64(query_image)
        for i, c in enumerate(concepts):
            frame = c['frame']
            encoded_frame = encode_image_in_base64(frame)
            concepts[i]['frame'] = encoded_frame

        response = json.dumps({
            "status": 200,
            "message": "OK",
            "data": concepts,
            "query_image_bytes": query_image_bytes
        })
        return response
    else:
        # update the index and the features
        ndd.update_index(videoids=post_data['videoids'])

        response = json.dumps({
            "status": 200,
            "message": "OK"
        })
        return response


if __name__ == '__main__':

    ndd = NearDuplicateDetection(features_root='../data')

    serve(app, host='0.0.0.0', port=9000)
