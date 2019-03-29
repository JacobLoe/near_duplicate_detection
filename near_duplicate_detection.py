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

from keras.applications.vgg16 import preprocess_input as pivgg16
from keras.applications.vgg19 import preprocess_input as pivgg19
from keras.applications.resnet50 import preprocess_input as pires
from keras.applications.inception_v3 import preprocess_input as piin3
from keras.applications.inception_resnet_v2 import preprocess_input as piin2

from keras.applications.densenet import preprocess_input as pidense

from keras.applications.nasnet import preprocess_input as pinas

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
def extract_features(model,frame,pi):

    frame=np.expand_dims(frame, axis=0)
    frame=pi(frame)#preprocess_input(frame)
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
    print('load models')
                                                                                                                                #lowest distance
#    NASNetMobile_model = NASNetMobile(weights='imagenet', include_top=False,pooling='max',input_shape=(224,224,3))             #64
#    DenseNet121_model = DenseNet121(weights='imagenet', include_top=False,pooling='max')                                       #80
#    DenseNet169_model = DenseNet169(weights='imagenet', include_top=False,pooling='max')                                       #102
#    DenseNet201_model = DenseNet201(weights='imagenet', include_top=False,pooling='max')                                       #83
#    vgg16_model = VGG16(weights='imagenet', include_top=False,pooling='max') #pooling applies global max pooling to last layer #448
#    vgg19_model = VGG19(weights='imagenet', include_top=False,pooling='max')                                                   #404
#    ResNet50_model = ResNet50(weights='imagenet', include_top=False,pooling='max')                                             #251
#    InceptionV3_model = InceptionV3(weights='imagenet', include_top=False,pooling='max')                                       #107
#    InceptionResNetV2_model = InceptionResNetV2(weights='imagenet', include_top=False,pooling='max')                           #77
#    NASNetLarge_model = NASNetLarge(weights='imagenet', include_top=False,pooling='max',input_shape=(331,331,3))               #152       #needs images as 331x331

    print('done')
    #####################################################################################################
'''    print('search for image in video')
    target_frame_features = extract_features(NASNetMobile_model,x, pinas)

    distances=[]

    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(NASNetMobile_model, frame, pinas)
            distances.append(euclidean(features,target_frame_features))

    #print(distances)
    print('\n','NASNetMobile ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(DenseNet121_model,x, pidense)

    distances=[]

    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(DenseNet121_model, frame, pidense)
            distances.append(euclidean(features,target_frame_features))

    #print(distances)
    print('\n','DenseNet121 ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(DenseNet169_model,x, pidense)

    distances=[]

    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(DenseNet169_model, frame, pidense)
            distances.append(euclidean(features,target_frame_features))

    #print(distances)
    print('\n','DenseNet169 ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(DenseNet201_model,x, pidense)

    distances=[]

    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(DenseNet201_model, frame, pidense)
            distances.append(euclidean(features,target_frame_features))

    #print(distances)
    print('\n','DenseNet201 ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(vgg16_model,x, pivgg16)

    distances=[]

    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(vgg16_model, frame, pivgg16)
            distances.append(euclidean(features,target_frame_features))

    #print(distances)
    print('\n','vgg16 ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(vgg19_model,x, pivgg19)

    distances=[]

    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(vgg19_model, frame, pivgg19)
            distances.append(euclidean(features,target_frame_features))

    #print(distances)
    print('\n','vgg19 ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(ResNet50_model,x, pires)

    distances=[]

    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(ResNet50_model, frame, pires)
            distances.append(euclidean(features,target_frame_features))

    #print(distances)
    print('\n','ResNet50_model ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(InceptionV3_model,x, piin3)

    distances=[]

    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(InceptionV3_model, frame, piin3)
            distances.append(euclidean(features,target_frame_features))

    #print(distances)
    print('\n','InceptionV3_model ',min(distances))
    #####################################################################################################
    target_frame_features = extract_features(InceptionResNetV2_model,x, piin2)

    distances=[]

    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(InceptionResNetV2_model, frame, piin2)
            distances.append(euclidean(features,target_frame_features))

    #print(distances)
    print('\n','InceptionResNetV2_model ',min(distances))'''
    #####################################################################################################
'''    print('search for image in video')
    target_frame_features = extract_features(NASNetLarge_model,x, pinas)

    distances=[]

    for begin_ms,end_ms in tqdm(shots):
        frame_list=read_video(args.video_path,begin_ms,end_ms)
        for frame in frame_list:
            features = extract_features(NASNetLarge_model, frame, pinas)
            distances.append(euclidean(features,target_frame_features))

    #print(distances)
    print('\n','NASNetLarge ',min(distances))'''
    #####################################################################################################
