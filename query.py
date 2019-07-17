import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import datetime
import glob

from keras.preprocessing import image

from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances

from extract_features import extract_features, load_model

#########################################################################################################
# save a frame to disk given a timestamp, the timestamp is expected to be the 'index' of a frame


def save_frame_to_disk(video_path, timestamp, img_name):
    vid = cv2.VideoCapture(video_path)
    vid.set(cv2.CAP_PROP_POS_FRAMES, timestamp)
    ret, frame = vid.read()
    frame = cv2.resize(frame, (299, 299))
    cv2.imwrite(img_name, frame)
########################################################################################################


def get_features(features_path, target_image):
    model = load_model()
    print('load target_image')
    target_img = image.load_img(target_image, target_size=(299, 299))
    target_img = image.img_to_array(target_img)
    print('done')

    print('extract feature for target image')
    target_feature = extract_features(model, target_img)
    print('done')

    source_video = []
    shot_begin_frame = []
    frame_timestamp = []
    frame_path = []

    list_features_path = glob.glob(os.path.join(features_path, '**/*.npy'), recursive=True)

    feature_list =[]
    for feature in tqdm(list_features_path, total=len(list_features_path)):
        fv = np.load(feature)
        feature_list.append(fv[0])  # create a list of features for the distance calculation

        i_ft = os.path.split(feature)  # the timestamp in ms of the feature
        i_sbf = os.path.split(i_ft[0])  # the timestamp in ms at which the shot starts where the feature is found
        i_sv = os.path.split(os.path.split(i_sbf[0])[0])  # the name of the videofile of the feature

        # recreate the path to the image from the path to the feature
        i_fp = os.path.join(i_sv[0], i_sv[1], 'frames', i_sbf[1], i_ft[1][:-4]+'.png')  # the path to image corresponding to the feature

        # add information for the frame
        source_video.append(i_sv[1])  # save the name of the source video
        shot_begin_frame.append(i_sbf[1])  # save the beginning timestamp of the shot the feature is from
        frame_timestamp.append(i_ft[1][:-4])  # save the specific timestamp the feature is at
        frame_path.append(i_fp)  # save the path of the feature

    features = {'feature_list': feature_list, 'target_feature': target_feature}
    info = {'source_video': source_video, 'shot_begin_frame': shot_begin_frame, 'frame_timestamp': frame_timestamp, 'frame_path': frame_path}
    return features, info
########################################################################################################


def write_to_html(lowest_distances, results_path, num_results, target_image):

    # FIX ME filter distances
    #filtered_distances = []
    #hits = 0
    #index = 0
    #while((hits < num_results) or (index < len(lowest_distances))): #repeat filtering until num_results results are found or there are no distances in the list anymore
    #    #if the current frame and the following frame are from the same video and the same shot
    #    if (lowest_distances[index][1] == lowest_distances[index+1][1]) and (lowest_distances[index][2] == lowest_distances[index+1][2]):
    #       index+=1
    #    else:
    #       filtered_distances.append(lowest_distances[index])
    #       hits+=1
    #       index+=1
    filtered_distances = lowest_distances[:num_results]

    lowest_distances = filtered_distances
    # write to html
    html_str = '<!DOCTYPE html><html lang="en"><table cellspacing="20"><tr><th>thumbnail</th><th>videofile</th><th>frame timestamp</th><th>distance</th></tr>'
    # add the target image to the html
    html_str += str('<tr><td><img src="{}"></td></tr>'.format(target_image))

    # append the found images to the html
    for dis, sv, sbf, ft, fp in lowest_distances:
        # convert ms timestamp to hh:mm:ss
        ft = str(datetime.timedelta(seconds=int(ft)/1000))

        # write an entry for the table, format is: frame_path, source_video, frame_timestamp, distance
        html_str += str('<tr><td><img src="{}"></td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(fp, sv, ft, dis))

    html_str += '</table></html>'
    aux = os.path.split(results_path)
    if len(aux[0]) == 0:
        with open(results_path, 'w') as f:
            f.write(html_str)
    else:
        os.makedirs(aux[0])
        with open(results_path, 'w') as f:
            f.write(html_str)
#########################################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", help="the directory in the feature vectors for the videos lie")
    parser.add_argument("target_image", help="the path to image that is to be searched for in the video")
    parser.add_argument("--results_path", nargs='?', default='results.html', help="filename for the results, the default name is results.html")
    parser.add_argument("--num_cores", type=int, default=4, help="specify the number cpu cores used for distance calculation, default value is 4")
    parser.add_argument("--num_results", type=int, default=30, help="specify how many frames are to be returned in the .html-file,default value is 30")
    args=parser.parse_args()

    features, info = get_features(args.features_dir, args.target_image)

    # calculate the distance for all features
    distances = pairwise_distances(features['feature_list'], features['target_feature'], metric=euclidean, n_jobs=args.num_cores)
    # sort by distance, ascending
    lowest_distances = sorted(zip(distances, info['source_video'], info['shot_begin_frame'], info['frame_timestamp'], info['frame_path']))

    write_to_html(lowest_distances, args.results_path, args.num_results, args.target_image)