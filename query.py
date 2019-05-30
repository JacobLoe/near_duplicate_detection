import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import datetime

from keras.preprocessing import image

from scipy.spatial.distance import euclidean

from extract_features import extract_features,load_model
#########################################################################################################
#save a frame to disk given a timestamp, the timestamp is expected to be the 'index' of a frame
def save_frame_to_disk(video_path,timestamp,img_name):
    vid = cv2.VideoCapture(video_path)
    vid.set(cv2.CAP_PROP_POS_FRAMES,timestamp)
    ret, frame = vid.read()
    frame=cv2.resize(frame,(224,224))
    cv2.imwrite(img_name,frame)
########################################################################################################
def main(features_path,target_image,results): 
   model=load_model()
   print('load target_image')
   target_img = image.load_img(target_image, target_size=(224, 224))
   target_img = image.img_to_array(target_img)
   print('done')

   print('extract features for target image')
   target_features = extract_features(model,target_img)
   print('done')
   #################################################
   distances=[]    
   source_video=[]
   shot_begin_frame=[]
   frame_timestamp=[]

   for video_dir in tqdm(os.listdir(features_path)):
       try:
           for shot in tqdm(os.listdir(features_path+video_dir+'/features')):
               for feature_vector in os.listdir(features_path+video_dir+'/features/'+shot):
                   fv=np.load(features_path+video_dir+'/features/'+shot+'/'+feature_vector)
                   distances.append(euclidean(fv,target_features))

                   #add information for the frame
                   source_video.append(video_dir)
                   shot_begin_frame.append(shot)
                   frame_timestamp.append(feature_vector[:-4])
       except:
           pass
   #sort by distance, ascending, take only the first 10 images
   lowest_distances=sorted(zip(distances,source_video,shot_begin_frame,frame_timestamp))[:9]

   #write to html
   html_str='<!DOCTYPE html><html lang="en"><table cellspacing="20"><tr><th>thumbnail</th><th>videofile</th><th>frame timestamp</th><th>distance</th></tr>'
   #add the target image to the html
   html_str+=str('<tr><td><img src="{}"></td></tr>'.format(target_image))

   #append the found images to the html
   for d,sv,sbf,ft in lowest_distances:
       img_path=features_path+sv+'/shots/'+sbf+'/'+ft+'.png'

       # convert ms timestamp to hh:mm:ss
       ft = str(datetime.timedelta(seconds=int(ft)/1000))

       html_str+=str('<tr><td><img src="{}"></td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(img_path,sv,ft,d))

   html_str+='</table></html>'
   with open(results,'w') as f:
        f.write(html_str)
#########################################################################################################
if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("features_dir",help="the directory in the feature vectors for the videos lie")
   parser.add_argument("target_image",help="the path to image that is to be searched for in the video")
   parser.add_argument("--results",nargs='?',default='results.html',help="filename for the results")
   args=parser.parse_args()

   main(args.features_dir,args.target_image,args.results)
