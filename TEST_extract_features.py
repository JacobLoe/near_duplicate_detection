import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import glob

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
#################################################################
# uses a given model and an image(numpy array) and returns its features


def extract_features(model, frame):
    print(np.shape(frame))
    frame = np.expand_dims(frame, axis=0)
    print(np.shape(frame))
    frame = preprocess_input(frame)
    features = model.predict(frame)
    features = features/np.linalg.norm(features)  # normalize length of feature vector

    return features
###################################################


def load_model():
    print('load model')
    model = InceptionResNetV2(weights='imagenet',input_shape=(299, 299, 3))
    model = Model(inputs=model.input,output=model.get_layer('avg_pool').output)
    print('done')
    return model
####################################################


def main(features_path):

    model_name = 'features_InceptionResNetV2_avgpoolLayer'

    images = os.path.join(features_path, '*/*/frames/*/*.png')
    list_images_path = glob.glob(images)  # get the list of videos in videos_dir
    cp = os.path.commonprefix(list_images_path)  # get the common dir between paths found with glob
   
    list_features_path = [os.path.split(                # split of the shots folder
                         os.path.split(                
                         os.path.split(                # split of the image-file name
                         os.path.join(features_path, os.path.relpath(p, cp))
                         )[0])[0])[0]
                         for p in list_images_path]

    model = load_model()

    batch_size = 64

    idg = ImageDataGenerator()
    di = DirectoryIterator(features_path, idg, batch_size=batch_size, class_mode=None, shuffle=False, target_size=(299, 299))
    aux_paths = []
    for i, paths in tqdm(enumerate(zip(list_images_path, list_features_path)), total=len(list_images_path)):
        aux_paths.append(paths)
        #print(aux_paths)
        #print(np.shape(di.next()))

        if( i % (batch_size-1) == 0):
            #print(np.shape(di.next()))
            #print(i)

            model.predict_on_batch(di.next())
            #break
        elif(i == (len(list_images_path)-1)):
            #print(np.shape(di.next()))
            print(i)
        #break
#########################################################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", help="the directory where the feature-vectors are to be stored, for example 'features'")
    args = parser.parse_args()

    main(args.features_dir)
