import os
import argparse
import numpy as np
from tqdm import tqdm
import glob
from PIL import Image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model
from idmapper import TSVIdMapper
import shutil

VERSION = '20200910'      # the version of the script
EXTRACTOR = 'features'
STANDALONE = False  # manages the creation of .done-files, if set to false no .done-files are created and the script will always overwrite old results


def extract_features(model, frame):
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    features = model.predict(frame)
    features = features / np.linalg.norm(features)  # normalize length of feature vector

    return features


def load_model():
    print('load model')
    model = InceptionResNetV2(weights='imagenet', input_shape=(299, 299, 3))
    model = Model(inputs=model.input, output=model.get_layer('avg_pool').output)
    print('finished loading model')
    return model


def extract_all_features_from_movie(features_path, file_extension, model, video_name):
    frames_path = os.path.join(os.path.split(features_path)[0], 'frames')
    # get the paths to all frames for the video
    frames_path = glob.glob(os.path.join(frames_path, '**', '*' + file_extension), recursive=True)
    if not frames_path:
        raise Exception('There were no images found with the file extension "{file_extension1}". '
                        'Check if the correct extension was used for the feature extraction or'
                        'consider running the image extraction for "{video_name} with "{file_extension2}" as the extension"'
                        ''.format(file_extension1=file_extension, file_extension2=file_extension, video_name=video_name))

    for fp in tqdm(frames_path):
        # open the image and extract the feature
        frame = Image.open(fp)
        frame = frame.convert('RGB')
        frame = frame.resize((299, 299))  # resize the image to the size used by inception
        feature = extract_features(model, frame)  # run the model on the frame

        # create the folder for the shot, to create the same structure as in the frames folder
        shot_folder_path = os.path.join(features_path, os.path.split(os.path.split(fp)[0])[1])
        if not os.path.isdir(shot_folder_path):
            os.mkdir(shot_folder_path)

        # get the name from the frame and save the feature with same name
        np_feature_name = os.path.split(fp)[1][:-len(file_extension)]
        feature_path = os.path.join(shot_folder_path, np_feature_name)
        np.save(feature_path, feature)


def main(features_root, file_extension, videoids, idmapper):
    model = load_model()
    # repeat until all movies are processed correctly
    for videoid in tqdm(videoids):
        try:
            video_rel_path = idmapper.get_filename(videoid)
        except KeyError as err:
            raise Exception("No such videoid: '{videoid}'".format(videoid=videoid))

        video_name = os.path.basename(video_rel_path)[:-4]
        features_dir = os.path.join(features_root, videoid, EXTRACTOR)

        # check .done-file of the image extraction for parameters to be added to the .done-file of the feature extraction
        # assume that the image extraction is set to STANDALONE if the feature extraction is, Otherwise this check will be skipped
        previous_parameters = ''
        try:
            if STANDALONE:
                with open(os.path.join(features_root, videoid, 'frames', '.done')) as done_file:
                    for i, line in enumerate(done_file):
                        if not i == 0:
                            previous_parameters = '\n' + previous_parameters + line
        except FileNotFoundError as err:
            raise Exception('The results of the image extraction for "{video_name}" cannot be found. The image extraction has to be run again'.format(video_name=video_name))

        done_file_path = os.path.join(features_dir, '.done')
        # create the version for a run, based on the script version and the used parameters
        done_version = VERSION+'\n'+file_extension+previous_parameters

        if not os.path.isfile(done_file_path) or not open(done_file_path, 'r').read() == done_version:
            print('feature extraction results missing or version did not match, extracting features for "{video_name}"'.format(video_name=video_name))
            # create the folder for the features, delete the old folder to prevent issues with older versions
            if not os.path.isdir(features_dir):
                os.makedirs(features_dir)
            else:
                shutil.rmtree(features_dir)
                os.makedirs(features_dir)

            extract_all_features_from_movie(features_dir, file_extension, model, video_name)
            # create a hidden file to signal that the feature extraction for a movie is done
            # write the current version of the script in the file
            if STANDALONE:
                with open(done_file_path, 'w') as d:
                    d.write(done_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", help="the directory where the feature-vectors are to be stored, for example 'features'")
    parser.add_argument('file_mappings', help='path to file mappings .tsv-file')
    parser.add_argument("videoids", help="List of video ids. If empty, entire corpus is iterated.", nargs='*')
    parser.add_argument("--file_extension", default='.jpeg', choices=('.jpeg', '.png'), help="use the extension in which the frames were saved, only .png and .jpg are supported, default is .jpeg")
    args = parser.parse_args()

    idmapper = TSVIdMapper(args.file_mappings)
    videoids = args.videoids if len(args.videoids) > 0 else parser.error('no videoids found')

    main(args.features_dir, args.file_extension, videoids, idmapper)
