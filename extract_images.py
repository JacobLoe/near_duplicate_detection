import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
#################################################################
def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root=tree.getroot().findall('content')
    info={}
    timestamps=[]
    for child in root[0].iter():
        if child.tag == 'shot':
            items=child.items()
            timestamps.append((int(items[3][1]),int(items[3][1])+int(items[1][1])-1)) #frames
            #timestamps.append((int(items[4][1]),int(items[4][1])+int(items[2][1])-1)) #ms
    return timestamps #in frames
#############################################################################################
#read video file frame by frame, beginning and ending with a timestamp
def save_shot_frames(video_path,frame_path,start_ms,end_ms):
    vid = cv2.VideoCapture(video_path)

    frame_size=(224, 224)
    #print(start_ms/1000,end_ms/1000)
    #print(range(int(end_ms/1000-start_ms/1000)+1))

#    for i in range(int(end_ms/1000-start_ms/1000)+1):
#        if not (start_ms/1000+i)==(int(end_ms/1000-start_ms/1000)+1):
#           vid.set(cv2.CAP_PROP_POS_MSEC,start_ms+i*1000)
#           ret, frame = vid.read()
#           frame=cv2.resize(frame,frame_size)
#           name = frame_path+str(start_ms+i)+'.png'
#           cv2.imwrite(name,frame)

    middle_frame=start_frame+round((end_ms-start_ms)/2)
    vid.set(cv2.CAP_PROP_POS_MSEC,middle_frame)
    ret, frame = vid.read()
    frame=cv2.resize(frame,frame_size)
    name = frame_path+str(middle_frame)+'.png'
    cv2.imwrite(name,frame)
#########################################################################################################
if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("videos_dir",help="the directory where the video-files are stored")
   parser.add_argument("features_dir",help="the directory where the images are to be stored")
   args=parser.parse_args()

   videos_path=args.videos_dir
   features_path=args.features_dir

   print('begin iterating through videos')
   for video_dir in tqdm(os.listdir(videos_path)):
       # create a new directory for each video
       if not os.path.isdir(features_path+video_dir):
          os.mkdir(features_path+video_dir)
       
       #get the shot timestamps generated by shotdetect
       shot_timestamps = read_shotdetect_xml(features_path+video_dir+'/shot_detection/result.xml')
       
       #save frames for each shot
       for video in os.listdir(videos_path+video_dir):
           # create a parent directory for all shots
           if not os.path.isdir(features_path+video_dir+'/shots/'):
              os.mkdir(features_path+video_dir+'/shots/')

           for start_frame,end_frame in tqdm(shot_timestamps):
               #create a dir for a specific shot, the name are the boundaries in ms 
               shot_path = features_path+video_dir+'/shots/'+str(start_frame)+'/'
               if not os.path.isdir(shot_path):
                  os.mkdir(shot_path)

               save_shot_frames(videos_path+video_dir+'/'+video,
                          shot_path,                          
                          start_frame,end_frame)
