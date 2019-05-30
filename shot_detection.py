import subprocess
import argparse
from tqdm import tqdm
import os
#########################################################################################################
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("videos_dir",help="the directory where the video-files are stored")
   parser.add_argument("features_dir",help="the directory where the images are to be stored")
   parser.add_argument("overwrite",type=bool,nargs='?',default=True,help="overwrite existing shot detections, default=True")
   args=parser.parse_args()

   videos_path=args.videos_dir
   features_path=args.features_dir

   if not os.path.isdir(features_path):
      os.mkdir(features_path)
   #########################################

   for video_dir in os.listdir(videos_path):
       try:
          if not os.path.isdir(features_path+video_dir):
             os.mkdir(features_path+video_dir)

          for video in os.listdir(videos_path+video_dir):
              if args.overwrite:
                 print('start shotdetection for {}'.format(video_dir))

                 if not os.path.isdir(features_path+video_dir+'/shot_detection'):
                    os.mkdir(features_path+video_dir+'/shot_detection')

                  keywords=['-i',videos_path+video_dir+'/'+video,
                           '-o',features_path+video_dir+'/shot_detection',
                           '-s','60']
                 process=['shotdetect']+keywords
                 p=subprocess.run(process,bufsize=0,
                                  shell=False,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
                 with open(features_path+video_dir+'/shot_detection/log.txt','w') as f:
                      f.write(str(p.stderr))
       except:
          pass
