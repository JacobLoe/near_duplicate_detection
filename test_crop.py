from PIL import Image, ImageChops
import numpy as np
from crop_image import trim

im0 = np.asarray(Image.open('test_images/171940.jpg'))[62:156, :, :]
im1 = np.asarray(Image.open('test_images/612240.jpg'))[:, 48:466, :]
im2 = np.asarray(Image.open('test_images/630360.jpg'))[:, 56:460, :]
im3 = np.asarray(Image.open('test_images/641720.jpg'))[:, 64:450, :]
im4 = np.asarray(Image.open('test_images/645040.jpg'))[:, 65:449, :]
im5 = np.asarray(Image.open('test_images/646240.jpg'))[:, 64:450, :]
im6 = np.asarray(Image.open('test_images/647920.jpg'))[:, 64:450, :]
im7 = np.asarray(Image.open('test_images/650320.jpg'))[:, 64:450, :]
im8 = np.asarray(Image.open('test_images/1226440.jpg'))[:, :257, :]     #relevant part is just in the left half
im9 = np.asarray(Image.open('test_images/1999170.jpg'))

i = 257
print(im8[219][i-1])
print(np.shape(im8))

Image.fromarray(im8).show()
