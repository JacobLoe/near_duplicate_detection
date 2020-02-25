from PIL import Image
import numpy as np
from crop_image import trim
from tqdm import tqdm

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
im10 = np.asarray(Image.open('test_images/216363_1.jpg'))   # no cropping required
im11 = np.asarray(Image.open('test_images/1646283.jpg'))[36:188, 47:400, :]
im12 = np.asarray(Image.open('test_images/2601643_1.jpg'))[72:, 54:461, :]
images_optimal_crop = [im0, im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11, im12]

images = [Image.open('test_images/172040.jpg'), Image.open('test_images/612240.jpg'), Image.open('test_images/630360.jpg'),
          Image.open('test_images/641720.jpg'), Image.open('test_images/645040.jpg'), Image.open('test_images/646240.jpg'),
          Image.open('test_images/647920.jpg'), Image.open('test_images/650320.jpg'), Image.open('test_images/1226440.jpg'),
          Image.open('test_images/1999320.jpg'), Image.open('test_images/216363_1.jpg'), Image.open('test_images/1646283.jpg'),
          Image.open('test_images/2601643_1.jpg')]
cc = {}
for t in tqdm(range(100)):
    c = []
    for i, image in enumerate(images):
        a, _ = trim(image, t)
        residuals = [np.shape(a)[j]-x for j, x in enumerate(np.shape(images_optimal_crop[i]))]
        c.append(residuals)
    cc[str(t)] = c
scores = {}
for key in cc:
    aux = 0
    for i in cc[key]:
        for j in i:
            aux += np.absolute(j)
    scores[key] = aux
    # print(key, cc[key])
# print(scores)
print(min(scores, key=scores.get))
