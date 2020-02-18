from PIL import Image, ImageChops
import numpy as np
import os
from tqdm import tqdm

cf_path = 'static/features_videos/movies/Ferguson_Charles_Inside_Job/frames_cropped'
ff_path = 'static/features_videos/movies/Ferguson_Charles_Inside_Job/full_frames'
fc_path = 'static/features_videos/movies/Ferguson_Charles_Inside_Job/frames_comparison'

for shot in tqdm(os.listdir(cf_path)):
    if not shot == '.done':
        o = os.path.join(cf_path, shot, shot)
        p = os.path.join(ff_path, shot, shot)
        # print(o)
        oo = Image.open(o+'.jpg')
        pp = Image.open(p+'.jpg')
        # print(np.shape(oo))
        # print(os.path.join(fc_path, shot)+'_0.jpg')
        oo.save(os.path.join(fc_path, shot)+'_0.jpg')
        pp.save(os.path.join(fc_path, shot)+'_1.jpg')


# print(os.path.join(cf_path, os.listdir(cf_path)[0], os.listdir(cf_path)[0])+'.jpg')
# copyfile(os.path.join(cf_path, os.listdir(cf_path)[0], os.listdir(cf_path)[0])+'.jpg',
#          os.path.join(fc_path, os.listdir(cf_path)[0]))
