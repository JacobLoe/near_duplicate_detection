from PIL import Image
import os
from tqdm import tqdm

cf_path = 'static/features_videos/movies/Ferguson_Charles_Inside_Job/frames_cropped'
ff_path = 'static/features_videos/movies/Ferguson_Charles_Inside_Job/full_frames'
fc_path = 'static/features_videos/movies/Ferguson_Charles_Inside_Job/frames_comparison'

if not os.path.isdir(fc_path):
    os.makedirs(fc_path)

for shot in tqdm(os.listdir(cf_path)):
    if not shot == '.done':
        o = os.path.join(cf_path, shot, shot)
        p = os.path.join(ff_path, shot, shot)
        oo = Image.open(o+'.jpeg')
        pp = Image.open(p+'.jpeg')
        oo.save(os.path.join(fc_path, shot)+'_0.jpeg')
        pp.save(os.path.join(fc_path, shot)+'_1.jpeg')