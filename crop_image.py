# https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil/10616717#10616717
from PIL import Image, ImageChops


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -12)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
        #return im.crop(bbox), bbox
