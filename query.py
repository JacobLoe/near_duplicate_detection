import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import datetime
import glob

from keras.preprocessing import image

from scipy.spatial.distance import euclidean

from extract_features import extract_features,load_model
#########################################################################################################
#save a frame to disk given a timestamp, the timestamp is expected to be the 'index' of a frame
def save_frame_to_disk(video_path,timestamp,img_name):
    vid = cv2.VideoCapture(video_path)
    vid.set(cv2.CAP_PROP_POS_FRAMES,timestamp)
    ret, frame = vid.read()
    frame=cv2.resize(frame,(299,299))
    cv2.imwrite(img_name,frame)
########################################################################################################
def main(features_path,target_image,results):
   model=load_model()
   print('load target_image')
   target_img = image.load_img(target_image, target_size=(299, 299))
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
   frame_path=[]

   list_features_path = glob.glob(os.path.join(features_path,'**/*.npy'),recursive=True) #get the list of videos in videos_dir

   for feature in tqdm(list_features_path):
       fv=np.load(feature)
       distances.append(euclidean(fv,target_features))
       i_ft = os.path.split(feature)
       i_sbf = os.path.split(i_ft[0])
       i_sv = os.path.split(os.path.split(i_sbf[0])[0])

       #recreate the path to image
       #aux = os.path.split(os.path.split(os.path.split(feature)[0])[0])[0]
       i_fp = os.path.join(i_sv[0],i_sv[1],'shots',i_sbf[1],i_ft[1][:-4]+'.png')

       #add information for the frame
       source_video.append(i_sv[1]) #save the name of the source video
       shot_begin_frame.append(i_sbf[1]) #save the beginning timestamp of the shot the feature is from
       frame_timestamp.append(i_ft[1][:-4]) #save the specific timestamp the feature is at
       frame_path.append(i_fp)

   #sort by distance, ascending, take only the first 10 images
   lowest_distances=sorted(zip(distances,source_video,shot_begin_frame,frame_timestamp,frame_path))[:30]
   ## FIX ME filter distances

   ######################################################################################
   #write to html
   html_str='<!DOCTYPE html><html lang="en"><table cellspacing="20"><tr><th>thumbnail</th><th>videofile</th><th>frame timestamp</th><th>distance</th></tr>'
   #add the target image to the html
   html_str+=str('<tr><td><img src="{}"></td></tr>'.format(target_image))

   #append the found images to the html
   for dis,sv,sbf,ft,fp in lowest_distances:
       # convert ms timestamp to hh:mm:ss
       ft = str(datetime.timedelta(seconds=int(ft)/1000))

       # write an entry for the table, format is: frame_path, source_video, frame_timestamp, distance
       html_str+=str('<tr><td><img src="{}"></td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(fp,sv,ft,dis))

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
