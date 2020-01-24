from PIL import Image, ImageChops
import numpy as np

def trim(im):
	#im.show()
	bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))#creates a new image in the color of the top left pixel 
	print('bg: ', np.asarray(bg)[0][0])
	#    bg.show(title='monocolored image')
	diff = ImageChops.difference(im, bg)#Returns the absolute value of the pixel-by-pixel difference between input image and the monocolored one
	print(np.asarray(diff)[0][0], np.asarray(diff)[298][298])
	diff.show(title='')
	diff = ImageChops.add(diff, diff, 2.0,5)
	print(np.asarray(diff)[0][0], np.asarray(diff)[298][298])
	#print('diff: ', np.asarray(diff)[0][0])
	diff.show()
	bbox = diff.getbbox()
	#bbox = im.getbox()
	print(bbox)
	if bbox:
		return im.crop(bbox)

#a = Image.open('../953000.png')
a = Image.open('../60733_full.jpg')
a.show()
print('image shape: ',np.shape(a))
p = trim(a)
p.show()
print('trimmed shape: ',np.shape(p))












#image_data = np.asarray(a)
#image_data_bw = image_data.max(axis=2)
#print('image_data_bw', image_data_bw)
#non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
#print('non_empty_columns', non_empty_columns)
#non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
#print('non_empty_rows ',non_empty_rows)
#cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
#print('cropBox',cropBox)

#image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

#new_image = Image.fromarray(image_data_new)
#new_image.show()


#p = np.asarray(p)
#y = int((np.shape(a)[0] - np.shape(p)[0])/2)
#x = int((np.shape(a)[1] - np.shape(p)[1])/2)
#print(x,y)
#frame = p[y:y+np.shape(p)[0], x:x+np.shape(p)[1], :]

#sp = Image.fromarray(frame)#
#p.show()
