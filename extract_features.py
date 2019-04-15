import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import preprocess_input
#################################################################
# uses a given model and an image(numpy array) and returns its features
def extract_features(model,frame):

    frame=np.expand_dims(frame, axis=0)
    frame=preprocess_input(frame)
    features = model.predict(frame)

    return features
#########################################################################################################
if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("features_dir",help="the directory where the images are to be stored, for example 'features'")
   args=parser.parse_args()

   # model
   print('load model')
   InceptionV3_model = InceptionV3(weights='imagenet', include_top=False,pooling='max',input_shape=(224,224,3))
   print('done')

   features_path=args.features_dir+'/'

   features_path='features_videos/'
   for video_dir in tqdm(os.listdir(features_path)):
       #create a directory for the features
       if not os.path.isdir(features_path+video_dir+'/features'):
          os.mkdir(features_path+video_dir+'/features')

       for shot in tqdm(os.listdir(features_path+video_dir+'/shots')):
           for image_name in os.listdir(features_path+video_dir+'/shots/'+shot):
           
               frame = cv2.imread(features_path+video_dir+'/shots/'+shot+'/'+image_name)
               feature = extract_features(InceptionV3_model,frame)
               
               #save features in the new directory, sorted by shots, named like the frame the features are from 
               if not os.path.isdir(features_path+video_dir+'/features/'+shot):
                  os.mkdir(features_path+video_dir+'/features/'+shot)
               path=features_path+video_dir+'/features/'+shot+'/'+image_name[:-4]
               np.save(path,feature)
