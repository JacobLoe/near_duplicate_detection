from PIL import Image, ImageChops
import numpy as np


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    print(np.shape(bg), bg)
    bg.show(title='first pixel color')
    im.show()
    diff = ImageChops.difference(im, bg)
    diff.show()
    diff = ImageChops.add(diff, diff, 2.0, -100)
    diff.show()
    bbox = diff.getbbox()
    print(bbox)
    if bbox:
        return im.crop(bbox)

a = Image.open('static/features_videos/unterordner/movies/Maltsev_Sem_Occupy_Wall_Street/frames/60733/60733.jpg')
print(np.shape(a))
trim(a)
