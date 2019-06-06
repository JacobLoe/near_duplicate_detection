import subprocess
import argparse
import os
import glob
#########################################################################################################
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("videos_dir",help="the directory where the video-files are stored")
   parser.add_argument("features_dir",help="the directory where the images are to be stored")
   args=parser.parse_args()

   list_videos_path = glob.glob(os.path.join(args.videos_dir,'**/*.mp4'),recursive=True) #get the list of videos in videos_dir

   cp = os.path.commonprefix(list_videos_path) #get the common dir between paths found with glob

   list_features_path = [os.path.join(os.path.split(                  # split of the name of the videofile from the newly created feature-path
                         os.path.join(args.features_dir, #join the feature direction and the path in which the video is saved
                         os.path.relpath(p,cp)))[0],'shot_detection')
                         for p in list_videos_path] #create a list of paths where all the data (shotdetection,frames,features) are saved to
   
   for v_path,f_path in zip(list_videos_path,list_features_path):
       if not os.path.isdir(f_path):
          os.makedirs(f_path) #create the dir for the shotdetection
       print('start shotdetection for {}'.format(os.path.split(v_path)[1]))

       keywords=['-i',v_path,
                 '-o',f_path,
                 '-s','60']
       process=['shotdetect']+keywords
       p=subprocess.run(process,bufsize=0,
                                  shell=False,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)

       log_file = os.path.join(f_path,'log.txt')
       with open(log_file,'w') as f:
            f.write(str(p.stderr))
