#####################################################
## libraries
#####################################################
import cv2
import numpy as np
import argparse
import zipfile
from tqdm import tqdm
import os
import xmltodict

from keras.preprocessing import image
from keras.applications.nasnet import NASNetMobile
from keras.applications.nasnet import preprocess_input

from scipy.spatial.distance import euclidean
#####################################################
## functions
#####################################################
# returns the timestamps of a given .azp-file in milliseconds
def get_shots(azp):
    zip_ref = zipfile.ZipFile(azp)
    inxmlstr = zip_ref.read('content.xml')
    doc = xmltodict.parse(inxmlstr)
    shots = []
    for a in doc['package']['annotations']['annotation']:
        if a['@type'] == '#Shot':
            begin_ms = int(a['millisecond-fragment']['@begin'])
            end_ms = int(a['millisecond-fragment']['@end'])

            shots.append((begin_ms, end_ms))
    return sorted(shots)
#############################################################################################
#read video file frame by frame, beginning and ending with a timestamp
def read_video(video_path,start_ms,end_ms):
    #resolution_height=int(round(resolution_width * 9/16))
    #resolution=(resolution_width,resolution_height)
    vid = cv2.VideoCapture(video_path)
    frames=[]
    fps = vid.get(cv2.CAP_PROP_FPS)

    start_frame=int((start_ms/1000.0)*fps)
    end_frame=int((end_ms/1000.0)*fps)

    middle_frame=start_frame+round((end_frame-start_frame)/2)
    frame_size=(224, 224)
    timestamps=[]
    vid_length=0
    with tqdm(total=end_frame-start_frame+1) as pbar: #init the progressbar,with max length of the given segment
        vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        while(vid.isOpened()):
            ret, frame = vid.read()
            if (vid_length+start_frame)==end_frame:
                #pbar.update(1)
                break

            frame=cv2.resize(frame,frame_size)
            frames.append(np.array(frame,dtype='float32'))
            timestamps.append(vid_length+start_frame)
            #pbar.update(1) #update the progressbar
            vid_length+=1 #increase the vid_length counter

############################################################
#            vid.set(cv2.CAP_PROP_POS_FRAMES,middle_frame-1)
#            ret, frame = vid.read()
#            frame=cv2.resize(frame,frame_size)
#            frames.append(np.array(frame,dtype='float32'))
#            timestamps.append(middle_frame-1)

#            vid.set(cv2.CAP_PROP_POS_FRAMES,middle_frame)
#            ret, frame = vid.read()
#            frame=cv2.resize(frame,frame_size)
#            frames.append(np.array(frame,dtype='float32'))
#            timestamps.append(middle_frame)

#            vid.set(cv2.CAP_PROP_POS_FRAMES,middle_frame+1)
#            ret, frame = vid.read()
#            frame=cv2.resize(frame,frame_size)
#            frames.append(np.array(frame,dtype='float32'))
#            timestamps.append(middle_frame+1)
    vid.release()
    cv2.destroyAllWindows()
    return frames,timestamps
#########################################################################################################
#save a frame to disk given a timestamp, the timestamp is expected to be the 'index' of a frame
def save_frame_to_disk(video_path,timestamp):
    vid = cv2.VideoCapture(video_path)
    vid.set(cv2.CAP_PROP_POS_FRAMES,timestamp)
    ret, frame = vid.read()
    cv2.imwrite('lowest_distance_frame.png',frame)
#########################################################################################################
# uses a given model and an image(numpy array) and returns its features
def extract_features(model,frame):

    frame=np.expand_dims(frame, axis=0)
    frame=preprocess_input(frame)
    features = model.predict(frame)

    return features
#########################################################################################################
if __name__ == "__main__":
    ## command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path",help="the path to the videofile")
    parser.add_argument("azp_path",help="the path to a azp-file")
    parser.add_argument("target_image",help="the path to image that is to be searched for in the video")
    args=parser.parse_args()
    ##############################################
    print('load target_image and shot timestamps')
    img = image.load_img(args.target_image, target_size=(224, 224))
    x = image.img_to_array(img)

    shots = get_shots(args.azp_path)
    print('done')
    #################################################
    # models
    print('load model')
    NASNetMobile_model = NASNetMobile(weights='imagenet', include_top=False,pooling='max',input_shape=(224,224,3))

    print('done')
    #####################################################################################################
    print('search for image in video')
    target_frame_features = extract_features(NASNetMobile_model,x)

    distances=[]    
    source_video=[]
    source_scene=[]
    shot_begin_ms=[]
    shot_end_ms=[]
    frame_timestamp=[]

    for begin_ms,end_ms in tqdm(shots):
        frame_list,timestamps=read_video(args.video_path,begin_ms,end_ms)
        for i,frame in enumerate(frame_list):
            features = extract_features(NASNetMobile_model, frame)
            distances.append(euclidean(features,target_frame_features))

            #add information for the frame
            source_video.append(args.video_path)
            source_scene.append(args.azp_path)
            shot_begin_ms.append(begin_ms)
            shot_end_ms.append(end_ms)
            frame_timestamp.append(timestamps[i])
    all_distances={'source_video':source_video,'source_scene':source_scene,'shot_begin_ms':shot_begin_ms,'shot_end_ms':shot_end_ms,'distances':distances,'frame_timestamp':frame_timestamp}
    lowest_distance_index=np.argmin(all_distances['distances'])
    
    print('distance: ',all_distances['distances'][lowest_distance_index],'\n',
          'source_video : ',all_distances['source_video'][lowest_distance_index],'\n',
          'source_scene: ',all_distances['source_scene'][lowest_distance_index],'\n',
          'shot_begin_ms: ',all_distances['shot_begin_ms'][lowest_distance_index],'\n',
          'shot_end_ms: ',all_distances['shot_end_ms'][lowest_distance_index],'\n',
          'frame_timestamp: ',all_distances['frame_timestamp'][lowest_distance_index])

    save_frame_to_disk(args.video_path,all_distances['frame_timestamp'][lowest_distance_index])
