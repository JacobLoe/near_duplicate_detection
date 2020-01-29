from PIL import Image, ImageChops
import numpy as np
from scipy.spatial.distance import euclidean

def trim(im):
	#im.show()
	bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))#creates a new image in the color of the top left pixel 
	# print('bg: ', np.asarray(bg)[0][0])
	#    bg.show(title='monocolored image')
	diff = ImageChops.difference(im, bg)#Returns the absolute value of the pixel-by-pixel difference between input image and the monocolored one
	# print(np.asarray(diff)[0][0], np.asarray(diff)[298][298])
	# diff.show(title='')
	diff = ImageChops.add(diff, diff, 2.0, -5)
	# print(np.asarray(diff)[0][0], np.asarray(diff)[298][298])
	#print('diff: ', np.asarray(diff)[0][0])
	# diff.show()
	bbox = diff.getbbox()
	#bbox = im.getbox()
	# print(bbox)
	if bbox:
		return im.crop(bbox)

c0 = Image.open('static/features_videos/movies/Ferguson_Charles_Inside_Job/frames/1526880/1526880.jpg')
c1 = Image.open('static/features_videos/movies/Ferguson_Charles_Inside_Job/frames/1526880/1527880.jpg')
c2 = Image.open('static/features_videos/movies/Ferguson_Charles_Inside_Job/frames/1526880/1528880.jpg')
c3 = Image.open('static/features_videos/movies/Ferguson_Charles_Inside_Job/frames/1526880/1529880.jpg')
# c0.show()
c = [c0,c1,c2,c3]
for i in c:
	i.show()
	print(np.shape(i))
	print(np.asarray(i)[0][0])
	print(np.asarray(i)[0][359])
	print(np.asarray(i)[219][0])
	print(np.asarray(i)[219][359])
	i = trim(i)
	i.show()
	print(np.shape(i))
	print(euclidean(np.shape(i)[1]/np.shape(i)[0],4/3),'4/3')
	print(euclidean(np.shape(i)[1]/np.shape(i)[0],16/9),'16/9')
	break

# print(np.shape(a), np.asarray(a)[0][0], np.asarray(a)[219][359])
# print(np.shape(b), np.asarray(b)[0][0], np.asarray(b)[219][359])
# print(np.shape(c))
# print(np.asarray(b)[0][0], np.asarray(b)[219][359])
#
# print(np.shape(a)[1]/np.shape(a)[0])
# print(4/3)
# print(16/9)
# print(euclidean(np.shape(a)[1]/np.shape(a)[0],4/3))
# print(euclidean(np.shape(a)[1]/np.shape(a)[0],16/9))


# a.show()
# print('image shape: ',np.shape(a))
# p = trim(a)
# p.show()
# print('trimmed shape: ',np.shape(p))
