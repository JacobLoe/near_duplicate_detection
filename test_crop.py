from PIL import Image, ImageChops
import numpy as np

def trim(im, threshold):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -threshold)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

im0 = np.asarray(Image.open('test_images/172040.jpg'))[62:156, :, :]
im1 = np.asarray(Image.open('test_images/612240.jpg'))[:, 48:466, :]
im2 = np.asarray(Image.open('test_images/630360.jpg'))[:, 56:460, :]
im3 = np.asarray(Image.open('test_images/641720.jpg'))[:, 64:450, :]
im4 = np.asarray(Image.open('test_images/645040.jpg'))[:, 65:449, :]
im5 = np.asarray(Image.open('test_images/646240.jpg'))[:, 64:450, :]
im6 = np.asarray(Image.open('test_images/647920.jpg'))[:, 64:450, :]
im7 = np.asarray(Image.open('test_images/650320.jpg'))[:, 64:450, :]
im8 = np.asarray(Image.open('test_images/1226440.jpg'))[:, :257, :]     #relevant part is just in the left half
im9 = np.asarray(Image.open('test_images/1999320.jpg'))[95:125, 165:349, :]
images_optimal_crop = [im0, im1, im2, im3, im4, im5, im6, im7, im8, im9]

images = [Image.open('test_images/172040.jpg'), Image.open('test_images/612240.jpg'), Image.open('test_images/630360.jpg'),
          Image.open('test_images/641720.jpg'), Image.open('test_images/645040.jpg'), Image.open('test_images/646240.jpg'),
          Image.open('test_images/647920.jpg'), Image.open('test_images/650320.jpg'), Image.open('test_images/1226440.jpg'),
          Image.open('test_images/1999320.jpg')]
cc = {}
for t in [10, 12, 15, 25]: #5,7,10,12,15,17,
    c = []
    for i, image in enumerate(images):

        # print('optimal crop shape', np.shape(images_optimal_crop[i]))
        a = trim(image, t)
        # print('trim shape', np.shape(a))
        residuals = [np.shape(a)[j]-x for j, x in enumerate(np.shape(images_optimal_crop[i]))]
        c.append(residuals)
        # print('residuals', residuals)
        # print('\n')
    cc[str(t)] = c
for key in cc:
    print(key, cc[key])
