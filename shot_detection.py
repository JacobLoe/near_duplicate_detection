import subprocess
import argparse
import os
import glob
import shutil

VERSION = '20200425'      # the version of the script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", help="the directory where the video-files are stored")
    parser.add_argument("features_dir", help="the directory where the images are to be stored")
    args = parser.parse_args()

    list_videos_path = glob.glob(os.path.join(args.videos_dir, '**/*.mp4'), recursive=True)  # get the list of videos in videos_dir

    cp = os.path.commonprefix(list_videos_path)  # get the common dir between paths found with glob

    list_features_path = [os.path.join(
                          os.path.join(args.features_dir,
                          os.path.relpath(p, cp))[:-4], 'shot_detection')  # add a new dir 'VIDEO_FILE_NAME/shot_detection' to the path
                          for p in list_videos_path]  # create a list of paths where all the data (shot-detection,frames,features) are saved to

    done = 0
    while done < len(list_features_path):  # repeat until all movies in the list have been processed correctly
        print('-------------------------------------------------------')
        for v_path, f_path in zip(list_videos_path, list_features_path):
            video_name = os.path.split(v_path)[1]
            done_file_path = os.path.join(f_path, '.done')
            # create a folder for the shot-detection results
            if not os.path.isdir(f_path):
                os.makedirs(f_path)
                print('start shotdetection for {}'.format(video_name))

                keywords = ['-i', v_path,
                            '-o', f_path,
                            '-s', '60']
                process = ['shotdetect']+keywords
                p = subprocess.run(process, bufsize=0,
                                   shell=False,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

                log_file = os.path.join(f_path, 'log.txt')
                with open(log_file, 'w') as f:
                    f.write(str(p.stderr))

                # create a hidden file to signal that the image-extraction for a movie is done
                # write the current version of the script in the file
                with open(done_file_path, 'a') as d:
                    d.write(VERSION)
                done += 1  # count the instances of the image-extraction done correctly
                # do nothing if a .done-file exists and the versions in the file and the script match
            elif os.path.isfile(done_file_path) and open(done_file_path, 'r').read() == VERSION:
                done += 1  # count the instances of the image-extraction done correctly
                print('shot-detection was already done for {}'.format(video_name))
                # if the folder already exists but the .done-file doesn't, delete the folder
            elif os.path.isfile(done_file_path) and not open(done_file_path, 'r').read() == VERSION:
                shutil.rmtree(f_path)
                print('versions did not match for {}'.format(video_name))
            elif not os.path.isfile(done_file_path):
                shutil.rmtree(f_path)
                print('shot-detection was not done correctly for {}'.format(video_name))