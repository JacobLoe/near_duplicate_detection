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

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201

from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

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
def read_video(video,start_ms,end_ms):
    #resolution_height=int(round(resolution_width * 9/16))
    #resolution=(resolution_width,resolution_height)
    vid = cv2.VideoCapture(video)
    frames=[]
    fps = vid.get(cv2.CAP_PROP_FPS)

    start_frame=int((start_ms/1000.0)*fps)
    end_frame=int((end_ms/1000.0)*fps)

    middle_frame=start_frame+round((end_frame-start_frame)/2)
    vid_length=0
    frame_size=(224, 224)
    with tqdm(total=end_frame-start_frame+1) as pbar: #init the progressbar,with max length of the given segment
#        if vid.isOpened():
#         while(vid.isOpened()):
#            ret, frame = vid.read() # if ret is false, frame has no content
#            if not ret:
#                break
#            if vid_length>=start_frame:
#                frame=cv2.resize(frame,(224,224))
#                frames.append(frame) #add the individual frames to a list
#                pbar.update(1) #update the progressbar
#            if vid_length==end_frame:
#                pbar.update(1)
#                break
#            vid_length+=1 #increase the vid_length counter
            vid.set(cv2.CAP_PROP_POS_FRAMES,middle_frame-1)
            ret, frame = vid.read()
            frame=cv2.resize(frame,frame_size)
            cv2.imwrite('key_frames/{fname}.png'.format(fname=middle_frame-1),frame)
            frames.append(np.array(frame,dtype='float32'))

            vid.set(cv2.CAP_PROP_POS_FRAMES,middle_frame)
            ret, frame = vid.read()
            frame=cv2.resize(frame,frame_size)
            cv2.imwrite('key_frames/{fname}.png'.format(fname=middle_frame),frame)
            frames.append(np.array(frame,dtype='float32'))

            vid.set(cv2.CAP_PROP_POS_FRAMES,middle_frame+1)
            ret, frame = vid.read()
            frame=cv2.resize(frame,frame_size)
            cv2.imwrite('key_frames/{fname}.png'.format(fname=middle_frame+1),frame)
            frames.append(np.array(frame,dtype='float32'))
    vid.release()
    cv2.destroyAllWindows()
    return frames
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
    args=parser.parse_args()
    ##############################################
    print('load images')
    img = image.load_img('Scene_12_shot_0_frame_0_Company_men.png', target_size=(224, 224))
    x = image.img_to_array(img)

    shots = get_shots(args.azp_path)
    print('done')
    #################################################
    # models
    print('load models')                                                                                                        #lowest distance
    vgg16_model = VGG16(weights='imagenet', include_top=True,pooling='max') #pooling applies global max pooling to last layer #0.7661318778991699
    vgg19_model = VGG19(weights='imagenet', include_top=True,pooling='max')                                                   #0.5507549047470093
    ResNet50_model = ResNet50(weights='imagenet', include_top=True,pooling='max')                                             #1.0022526979446411
    InceptionV3_model = InceptionV3(weights='imagenet', include_top=True,pooling='max')                                       #8.319439580769543e-30
    InceptionResNetV2_model = InceptionResNetV2(weights='imagenet', include_top=True,pooling='max')                           #2.463427778351537e-35
    DenseNet121_model = DenseNet121(weights='imagenet', include_top=True,pooling='max')                                       #0.03612004593014717
    DenseNet169_model = DenseNet169(weights='imagenet', include_top=True,pooling='max')                                       #1.1113868951797485
    DenseNet201_model = DenseNet201(weights='imagenet', include_top=True,pooling='max')                                       #0.12311811745166779
#    NASNetLarge_model = NASNetLarge(weights='imagenet', include_top=True,pooling='max',input_shape=(331,331,3))               #152       #needs images as 331x331
#    NASNetMobile_model = NASNetMobile(weights='imagenet', include_top=True,pooling='max',input_shape=(224,224,3))             #64
    print('done')
    #####################################################################################################
#    print('search for image in video')
#    target_frame_features = extract_features(NASNetLarge_model,x)
#    distances=[]
#    for begin_ms,end_ms in tqdm(shots):
#        frame_list=read_video(args.video_path,begin_ms,end_ms)
#        for frame in frame_list:
#            features = extract_features(NASNetLarge_model, frame)
#            distances.append(euclidean(features,target_frame_features))
#    #print(distances)
#    print('\n','NASNetLarge ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(DenseNet121_model,x)
    distances=[]
    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(DenseNet121_model, frame)
            distances.append(euclidean(features,target_frame_features))
    #print(distances)
    print('\n','DenseNet121 ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(DenseNet169_model,x)
    distances=[]
    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(DenseNet169_model, frame)
            distances.append(euclidean(features,target_frame_features))
    #print(distances)
    print('\n','DenseNet169 ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(DenseNet201_model,x)
    distances=[]
    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(DenseNet201_model, frame)
            distances.append(euclidean(features,target_frame_features))
    #print(distances)
    print('\n','DenseNet201 ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(vgg16_model,x)
    distances=[]
    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(vgg16_model, frame)
            distances.append(euclidean(features,target_frame_features))
    #print(distances)
    print('\n','vgg16 ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(vgg19_model,x)
    distances=[]
    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(vgg19_model, frame)
            distances.append(euclidean(features,target_frame_features))
    #print(distances)
    print('\n','vgg19 ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(ResNet50_model,x)
    distances=[]
    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(ResNet50_model, frame)
            distances.append(euclidean(features,target_frame_features))
    #print(distances)
    print('\n','ResNet50_model ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(InceptionV3_model,x)
    distances=[]
    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(InceptionV3_model, frame)
            distances.append(euclidean(features,target_frame_features))
    #print(distances)
    print('\n','InceptionV3_model ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(InceptionResNetV2_model,x)
    distances=[]
    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(InceptionResNetV2_model, frame)
            distances.append(euclidean(features,target_frame_features))
    #print(distances)
    print('\n','InceptionResNetV2_model ',min(distances))
