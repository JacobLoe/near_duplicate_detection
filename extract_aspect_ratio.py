import os
import argparse
from tqdm import tqdm
import glob
import shutil
import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean

def save_aspect_ratio_to_csv(mrps, frames_dir, shot_timestamps):
    # ar = [res[0]/(res[1]) if not res[1] == 0 else 0 for res in mrps.values()]
    tu_ar_float = [21.01/9, 21/9, 16/9, 4/3, 1/1, 3/4, 9/16, 9/16.01]
    tu_ar_str = ['>21/9', '21/9', '16/9', '4/3', '1/1', '3/4', '9/16', '<9/16']
    # [2.3344444444444448, 2.3333333333333335, 1.7777777777777777, 1.3333333333333333, 1.0, 0.75, 0.5625, 0.5621486570893192]
    # ar_csv_path = os.path.join(frames_dir, 'aspect_ratio_per_shot.csv')
    # with open(ar_csv_path, 'w', newline='') as f:
    #     for i, key in enumerate(mrps):
    #         # convert the resolution of the shot to an aspect ratio
    #         # set 0 if resolution is 0
    #         if not mrps[key][1] == 0:
    #             ar = mrps[key][0]/mrps[key][1]
    #         else:
    #             ar = 0
    #
    #         # calculate the distance of the shot aspect ratio to the ratios in the list
    #         dist = [euclidean(ar, x) for x in tu_ar_float]
    #         # print(ar, mrps[key][0], mrps[key][1], dist, np.argmin(dist), tu_ar_str[np.argmin(dist)])
    #         line = str(key)+' '+str(shot_timestamps[i][1])+' '+str(tu_ar_str[np.argmin(dist)])
    #         f.write(line)
    #         f.write('\n')


def main(features_path, file_extension):
    print('begin iterating through videos')

    list_features_path = glob.glob(os.path.join(features_path, '**/frames'), recursive=True)

    done = 0
    while done < len(list_features_path):  # repeat until all movies in the list have been processed correctly
        print('-------------------------------------------------------')
        for f_path in tqdm(list_features_path, total=len(list_features_path)):
            done = save_aspect_ratio_to_csv(f_path, file_extension, done)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    parser.add_argument("--file_extension", default='.jpg', choices=('.jpg', '.png'), help="define the file-extension of the frames, only .png and .jpg are supported, default is .jpg")
    args = parser.parse_args()

    main(args.features_dir, args.file_extension)