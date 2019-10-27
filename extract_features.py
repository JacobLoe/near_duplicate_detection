import os
import argparse
import numpy as np
from tqdm import tqdm
import glob
import cv2
from PIL import Image
from crop_image import trim
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model

#################################################################
# uses a given model and an image(numpy array) and returns its features


def extract_features(model, frame):
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    features = model.predict(frame)
    features = features/np.linalg.norm(features)  # normalize length of feature vector

    return features
###################################################


def load_model():
    print('load model')
    model = InceptionResNetV2(weights='imagenet', input_shape=(299, 299, 3))
    model = Model(inputs=model.input, output=model.get_layer('avg_pool').output)
    print('finished loading model')
    return model
####################################################


def main(features_path):

    model_name = 'features_InceptionResNetV2_avgpoolLayer'

    images = os.path.join(features_path,  '**/frames/*/*.png')
    list_images_path = glob.glob(images, recursive=True)  # get the list of videos in videos_dir
    cp = os.path.commonprefix(list_images_path)  # get the common dir between paths found with glob
   
    list_features_path = [os.path.split(                # split of the shots folder
                          os.path.split(
                            os.path.split(                # split of the image-file name
                                os.path.join(features_path, os.path.relpath(p, cp))
                            )[0])[0])[0]
                          for p in list_images_path]
    model = load_model()
    done = 0
    while done < len(list_features_path):  # repeat until all frames in the list have been processed correctly
        if done != 0:
            done = 0
        print('-------------------------------------------------------')
        aux_frame_size = (0, 0)
        for i_path, f_path in tqdm(zip(list_images_path, list_features_path), total=len(list_images_path)):
            feature_name = os.path.split(i_path)[1][:-4]  # get the name of the image, remove the file extension

            shot = os.path.split(os.path.split(i_path)[0])[1]  # get the name of the shot for the image
            fp = os.path.join(f_path, model_name, shot)
            path = os.path.join(fp, feature_name)  # specify the path to which the feature is saved, the name is the same as the image (w/o the file-extension)
            path = path + '.npy'
            done_name = '.done' + feature_name
            done_path = os.path.join(fp, done_name)

            if not os.path.isfile(path) and not os.path.isfile(done_path):  # if neither the feature nor a .done-file exist, start extracting the feature
                if not os.path.isdir(fp):  # create the directory to save the features
                    os.makedirs(fp)

                # frame = cv2.imread(i_path)  # read the frame from disc
                frame = Image.open(i_path)
                frame = frame.convert('RGB')
                if args.crop_letterbox:
                    frame = trim(frame)
                if np.shape(frame) != aux_frame_size:
                    pass

                frame = frame.resize((299, 299))  # resize the image to the size used by inception

                feature = extract_features(model, frame)  # run the model on the frame
                np.save(path, feature)  # save the feature to disc
                # create a hidden file to signal that the feature-extraction for a frame is done
                open(done_path, 'a').close()
                done += 1  # count the instances of the feature-extraction done correctly
            elif os.path.isfile(path) and os.path.isfile(done_path):    # if both files exist, do nothing
                done += 1  # count the instances of the feature-extraction done correctly
            # if either the feature or the .done-file don't exist, something went wrong
            # the other file is deleted, so the process can be finished in the second iteration
            elif os.path.isfile(path) and not os.path.isfile(done_path):
                os.remove(path)
            elif not os.path.isfile(path) and os.path.isfile(done_path):
                os.remove(done_path)
        print('feature-extraction was already done for {}/{} features'.format(done, len(list_features_path)))
#########################################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", help="the directory where the feature-vectors are to be stored, for example 'features'")
    parser.add_argument("--crop_letterbox", type=bool, default=False, help="remove the letterbox of an image, takes a bool, default is False")
    args = parser.parse_args()

    main(args.features_dir)
