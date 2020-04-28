import os
import argparse
import numpy as np
from tqdm import tqdm
import glob
import shutil
from PIL import Image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model


VERSION = '20200425'      # the version of the script


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


def extract_all_features_from_movie(f_path, file_extension, done, model):
    video_name = os.path.split(f_path)[1]
    model_name = 'features_InceptionResNetV2_avgpoolLayer'
    features_dir = os.path.join(f_path, model_name)

    done_file_path = os.path.join(features_dir, '.done')

    if not os.path.isdir(features_dir):
        print('extracting features for {}'.format(video_name))
        os.makedirs(features_dir)
        # get the paths to all frames for the video
        frames_path = glob.glob(os.path.join(f_path, '**', '*' + file_extension), recursive=True)
        for frame_p in tqdm(frames_path):
            # open the image and extract the feature
            frame = Image.open(frame_p)
            frame = frame.convert('RGB')
            frame = frame.resize((299, 299))  # resize the image to the size used by inception
            feature = extract_features(model, frame)  # run the model on the frame

            # create the folder for the shot, to create the same structure as in the frames folder
            shot_folder_path = os.path.join(features_dir, os.path.split(os.path.split(frame_p)[0])[1])
            if not os.path.isdir(shot_folder_path):
                os.mkdir(shot_folder_path)

            # get the name from the frame and save the feature with same name
            np_feature_name = os.path.split(frame_p)[1][:-len(file_extension)]
            feature_path = os.path.join(shot_folder_path, np_feature_name)
            np.save(feature_path, feature)

        # create a hidden file to signal that the image-extraction for a movie is done
        # write the current version of the script in the file
        with open(done_file_path, 'a') as d:
            d.write(VERSION)
        done += 1  # count the instances of the image-extraction done correctly
    # do nothing if a .done-file exists and the versions in the file and the script match
    elif os.path.isfile(done_file_path) and open(done_file_path, 'r').read() == VERSION:
        done += 1  # count the instances of the image-extraction done correctly
        print('feature-extraction was already done for {}'.format(video_name))
    # if the folder already exists but the .done-file doesn't, delete the folder
    elif os.path.isfile(done_file_path) and not open(done_file_path, 'r').read() == VERSION:
        shutil.rmtree(features_dir)
        print('versions did not match for {}'.format(video_name))
    elif not os.path.isfile(done_file_path):
        shutil.rmtree(features_dir)
        print('feature-extraction was not done correctly for {}'.format(video_name))

    return done


def main(features_dir, file_extension):
    # get the directories of all the movies by searching for the frames folder
    features_dir = glob.glob(os.path.join(features_dir, '**/frames'), recursive=True)
    # cut off the frames folder from every path
    features_dir = [os.path.split(f)[0] for f in features_dir]
    done = 0
    model = load_model()
    while done < len(features_dir):  # repeat until all movies in the list have been processed correctly
        print('-------------------------------------------------------')
        for f_d in features_dir:
            done = extract_all_features_from_movie(f_d, file_extension, done, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", help="the directory where the feature-vectors are to be stored, for example 'features'")
    parser.add_argument("--file_extension", default='.jpeg', choices=('.jpeg', '.png'), help="use the extension in which the frames were saved, only .png and .jpg are supported, default is .jpeg")
    args = parser.parse_args()

    main(args.features_dir, args.file_extension)
