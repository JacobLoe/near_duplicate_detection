import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import preprocess_input

from scipy.spatial.distance import euclidean
#########################################################################################################
#save a frame to disk given a timestamp, the timestamp is expected to be the 'index' of a frame
def save_frame_to_disk(video_path,timestamp,img_name):
    vid = cv2.VideoCapture(video_path)
    vid.set(cv2.CAP_PROP_POS_FRAMES,timestamp)
    ret, frame = vid.read()
    frame=cv2.resize(frame,(224,224))
    cv2.imwrite(img_name,frame)
#########################################################################################################
# uses a given model and an image(numpy array) and returns its features
def extract_features(model,frame):

    frame=np.expand_dims(frame, axis=0)
    frame=preprocess_input(frame)
    features = model.predict(frame)

    return features
#########################################################################################################

if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("features_dir",help="the directory in the feature vectors for the videos lie")
   parser.add_argument("target_image",help="the path to image that is to be searched for in the video")

   args=parser.parse_args()

   features_path=args.features_dir
   ##############################################
   print('load model')
   InceptionV3_model = InceptionV3(weights='imagenet', include_top=False,pooling='max',input_shape=(224,224,3))
   print('done')

   print('load target_image')
   target_img = image.load_img(args.target_image, target_size=(224, 224))
   target_img = image.img_to_array(target_img)
   print('done')

   print('extract features for target image')
   target_features = extract_features(InceptionV3_model,target_img)
   print('done')
   #################################################

   distances=[]    
   source_video=[]
   shot_begin_frame=[]
   frame_timestamp=[]

   for video_dir in tqdm(os.listdir(features_path)):
       for shot in tqdm(os.listdir(features_path+video_dir+'/features')):
           for feature_vector in os.listdir(features_path+video_dir+'/features/'+shot):
               fv=np.load(features_path+video_dir+'/features/'+shot+'/'+feature_vector)
               distances.append(euclidean(fv,target_features))

               #add information for the frame
               source_video.append(video_dir)
               shot_begin_frame.append(shot)
               frame_timestamp.append(feature_vector[:-4])
   #sort by distance, ascending, take only the first 10 images
   lowest_distances=sorted(zip(distances,source_video,shot_begin_frame,frame_timestamp))[:9]

   #write to html
   html_str='<!DOCTYPE html><html lang="en"><table><tr><th>thumbnail</th><th>videofile</th><th>frame timestamp</th><th>distance</th></tr>'
   #add the target image to the html
   html_str+=str('<tr><td><img src="{}"></td></tr>'.format(args.target_image))

   #append the found images to the html
   for d,sv,sbf,ft in lowest_distances:
       img_path=features_path+sv+'/shots/'+sbf+'/'+ft+'.png'
       #video_path='../videos/'+sv+'/'+sv+'.mp4'
       #img_name=ft+'.png'
       #save_frame_to_disk(video_path,int(ft),img_name)
       html_str+=str('<tr><td><img src="{}"></td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(img_path,sv,ft,d))

   html_str+='</table></html>'
   with open('results.html','w') as f:
        f.write(html_str)
