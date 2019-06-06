import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import glob

from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
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
   model = InceptionResNetV2(weights='imagenet',input_shape=(299,299,3))
   model = Model(inputs=model.input,output=model.get_layer('avg_pool').output)
   print('done')
   return model
####################################################
def main(features_path):

   images = os.path.join(features_path,'*/*/shots/*/*.png')
   list_images_path = glob.glob(images) #get the list of videos in videos_dir
   #print(list_images_path)
   cp = os.path.commonprefix(list_images_path) #get the common dir between paths found with glob
   
   list_features_path = [os.path.split(                # split of the shots folder
                         os.path.split(                
                         os.path.split(                # split of the image-file name
                         os.path.join(features_path,os.path.relpath(p,cp))
                         )[0])[0])[0]
                         for p in list_images_path]
   model=load_model()
   for i_path,f_path in tqdm(zip(list_images_path,list_features_path),total=len(list_images_path)):

       feature_name = os.path.split(i_path)[1][:-4] #get the name of the image, remove the file extension
       shot = os.path.split(os.path.split(i_path)[0])[1] #get the name of the shot for the image

       fp = os.path.join(f_path,'features',shot)
       if not os.path.isdir(fp): #create the directory to save the features
          os.makedirs(fp)
       
       frame = cv2.imread(i_path) #read the image from disc
       feature = extract_features(model,frame) #run the model on the image
       path = os.path.join(fp,feature_name) #specify the path to which the feature is saved, the name is the same as the image (w/o the file-extension)
       np.save(path,feature) #save the feature to disc
#########################################################################################################
if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("features_dir",help="the directory where the feature-vectors are to be stored, for example 'features'")
   args=parser.parse_args()

   main(args.features_dir)
