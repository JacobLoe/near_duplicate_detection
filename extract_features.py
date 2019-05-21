import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from keras.preprocessing import image
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_v3 import preprocess_input
#from keras.applications.resnet50 import ResNet50
#from keras.applications.resnet50 import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
#from keras.applications.vgg19 import VGG19
#from keras.applications.vgg19 import preprocess_input
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg16 import preprocess_input
from keras.models import Model
#################################################################
# uses a given model and an image(numpy array) and returns its features
def extract_features(model,frame):

    frame=np.expand_dims(frame, axis=0)
    frame=preprocess_input(frame)
    features = model.predict(frame)
    features=features/np.linalg.norm(features) #normalize length of feature vector

    return features
###################################################
def load_model():
   print('load model')
   model = InceptionResNetV2(weights='imagenet',input_shape=(224,224,3))
   model = Model(inputs=model.input,output=model.get_layer('avg_pool').output)
   print('done')
   return model
####################################################
def main(features_path):

   model=load_model()
   for video_dir in tqdm(os.listdir(features_path)):
       #create a directory for the features
       if not os.path.isdir(features_path+video_dir+'/features'):
          os.mkdir(features_path+video_dir+'/features')

       for shot in tqdm(os.listdir(features_path+video_dir+'/shots')):
           for image_name in os.listdir(features_path+video_dir+'/shots/'+shot):
           
               frame = cv2.imread(features_path+video_dir+'/shots/'+shot+'/'+image_name)
               feature = extract_features(model,frame)
               #save features in the new directory, sorted by shots, named like the frame the features are from 
               if not os.path.isdir(features_path+video_dir+'/features/'+shot):
                  os.mkdir(features_path+video_dir+'/features/'+shot)
               path=features_path+video_dir+'/features/'+shot+'/'+image_name[:-4]
               np.save(path,feature)
#########################################################################################################
if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("features_dir",help="the directory where the feature-vectors are to be stored, for example 'features'")
   args=parser.parse_args()

   main(args.features_dir)
