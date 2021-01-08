import os
import argparse
import numpy as np
from tqdm import tqdm
import glob
from PIL import Image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.models import Model
import shutil
import logging

VERSION = '20210107'      # the date the script was last changed
EXTRACTOR = 'features'
STANDALONE = True  # manages the creation of .done-files, if set to false no .done-files are created and the script will always overwrite old results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False    # prevent log messages from appearing twice


def extract_features(model, frame):
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    features = model.predict(frame)
    features = features / np.linalg.norm(features)  # normalize length of feature vector

    return features


def load_model():
    model = InceptionResNetV2(weights='imagenet', input_shape=(299, 299, 3))
    model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    return model


def extract_all_features_from_movie(features_dir, file_extension, model):
    frames_path = os.path.join(os.path.split(features_dir)[0], 'frames')
    # get the paths to all frames for the video
    frames_path = glob.glob(os.path.join(frames_path, '*' + file_extension))
    if not frames_path:
        raise Exception('There were no images found with the file extension "{file_extension1}". '
                        'Check if the correct extension was used for the feature extraction or '
                        'if the image extraction was run with "{file_extension2}" as the extension"'
                        ''.format(file_extension1=file_extension, file_extension2=file_extension))

    for fp in tqdm(frames_path):
        # open the image and extract the feature
        frame = Image.open(fp)
        frame = frame.convert('RGB')
        frame = frame.resize((299, 299))  # resize the image to the size used by inception
        feature = extract_features(model, frame)  # run the model on the frame

        # get the name from the frame and save the feature with same name
        np_feature_name = os.path.split(fp)[1][:-len(file_extension)]
        feature_path = os.path.join(features_dir, np_feature_name)
        np.save(feature_path, feature)


def main(features_root, file_extension, videoids, force_run):
    model = load_model()
    # repeat until all movies are processed correctly
    for videoid in tqdm(videoids):

        features_dir = os.path.join(features_root, videoid, EXTRACTOR)

        # FIXME something is wrong with the .done-file
        # check .done-file of the image extraction for parameters to be added to the .done-file of the feature extraction
        # assume that the image extraction is set to STANDALONE if the feature extraction is, Otherwise this check will be skipped
        previous_parameters = ''
        try:
            if STANDALONE:
                with open(os.path.join(features_root, videoid, 'frames', '.done')) as done_file:
                    for i, line in enumerate(done_file):
                        if not i == 0:
                            previous_parameters += line
        except FileNotFoundError as err:
            raise Exception('The results of the image extraction cannot be found. The image extraction has to be run again')

        done_file_path = os.path.join(features_dir, '.done')
        # create the version for a run, based on the script version and the used parameters
        done_version = VERSION+'\n'+file_extension+'\n'+previous_parameters

        if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == done_version or force_run:
            logger.info('feature extraction results missing or version did not match, extracting features')
            # create the folder for the features, delete the old folder to prevent issues with older versions
            if not os.path.isdir(features_dir):
                os.makedirs(features_dir)
            else:
                shutil.rmtree(features_dir)
                os.makedirs(features_dir)

            extract_all_features_from_movie(features_dir, file_extension, model)
            # create a hidden file to signal that the feature extraction for a movie is done
            # write the current version of the script in the file
            if STANDALONE:
                with open(done_file_path, 'w') as d:
                    d.write(done_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", help="the directory where the feature-vectors are to be stored, for example 'features'")
    parser.add_argument("videoids", help="List of video ids. If empty, entire corpus is iterated.", nargs='*')
    parser.add_argument("--file_extension", default='.jpeg', choices=('.jpeg', '.png'), help="use the extension in which the frames were saved, only .png and .jpg are supported, default is .jpeg")
    parser.add_argument("--force_run", default=False, type=bool, help='sets whether the script runs regardless of the version of .done-files')
    args = parser.parse_args()

    main(args.features_dir, args.file_extension, args.videoids, args.force_run)
