import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import glob
import time
import shutil
#################################################################
def read_shotdetect_xml(path):
    tree = ET.parse(path)
    root=tree.getroot().findall('content')
    info={}
    timestamps=[]
    for child in root[0].iter():
        if child.tag == 'shot':
            items=child.items()
            timestamps.append((int(items[4][1]),int(items[4][1])+int(items[2][1])-1)) #ms
    return timestamps #in ms
#############################################################################################
#read video file frame by frame, beginning and ending with a timestamp
def save_shot_frames(video_path,frame_path,start_ms,end_ms):
    vid = cv2.VideoCapture(video_path)

    frame_size=(299, 299)

    for i in range(int(end_ms/1000-start_ms/1000)+1):
        if not (start_ms/1000+i)==(int(end_ms/1000-start_ms/1000)+1):
           vid.set(cv2.CAP_PROP_POS_MSEC,start_ms+i*1000)
           ret, frame = vid.read()
           frame=cv2.resize(frame,frame_size)
           name = os.path.join(frame_path,(str(start_ms+i*1000)+'.png'))
           cv2.imwrite(name,frame)
#########################################################################################################
def main(videos_path,features_path):
    print('begin iterating through videos')

    list_videos_path = glob.glob(os.path.join(videos_path,'**/*.mp4'),recursive=True) #get the list of videos in videos_dir

    cp = os.path.commonprefix(list_videos_path) #get the common dir between paths found with glob

    list_features_path = [os.path.join(                 
                          os.path.join(args.features_dir,               
                          os.path.relpath(p,cp))[:-4]) #add a new dir 'VIDEO_FILE_NAME/shot_detection' to the path
                          for p in list_videos_path] #create a list of paths where all the data (shotdetection,frames,features) are saved to
    
    for v_path,f_path in tqdm(zip(list_videos_path,list_features_path),total=len(list_videos_path)):
        vid = cv2.VideoCapture(v_path)
        #get the shot timestamps generated by shotdetect
        shot_timestamps = read_shotdetect_xml(os.path.join(f_path,'shot_detection/result.xml'))
        print(shot_timestamps[0])
        #start_ms = shot_timestamps[0][0]
        #end_ms = shot_timestamps[0][1]
        list_shot_ts_ms = [[start_ms + i*1000 for i in range(int(end_ms/1000-start_ms/1000)+1) if not (start_ms/1000+i)==(int(end_ms/1000-start_ms/1000)+1)] for start_ms, end_ms in shot_timestamps]

        print(np.shape(list_shot_ts_ms))
        print(list_shot_ts_ms[0])
        print(list_shot_ts_ms[1])
        print(list_shot_ts_ms[2])

        for shot_ts_ms in list_shot_ts_ms:
            #create a dir for a specific shot, the name are the boundaries in ms 
            frames_path = os.path.join(f_path,'frames',str(shot_ts_ms[0]))
            if not os.path.isdir(frames_path):
               os.makedirs(frames_path)
            for i in
            break

        break
'''       if os.path.isdir(os.path.join(f_path,'frames')): #remove the old directory
          shutil.rmtree(os.path.join(f_path,'frames'))
      
       for start_frame,end_frame in tqdm(shot_timestamps):

           save_shot_frames(v_path,
                            frames_path,                          
                            start_frame,end_frame)
'''
#########################################################################################################
if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("videos_dir",help="the directory where the video-files are stored")
   parser.add_argument("features_dir",help="the directory where the images are to be stored")
   args=parser.parse_args()

   main(args.videos_dir,args.features_dir)
