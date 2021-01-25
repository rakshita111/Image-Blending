
#GAUSSIAN AND LAPLACIAN PYRAMID IMPLEMENTATION FOR COLOR IMAGES OF ANY DIMENSIONS OF THE IMAGE

import cv2
import numpy as np
import  math
from scipy import signal
from skimage.exposure import rescale_intensity

#defining function for Gaussian kernel
def get_gauss_kernel(size,sigma):
    center=int(size/2)
    kernel=np.zeros((size,size))
    for i in range(size):
       for j in range(size):
          diff=np.sqrt((i-center)**2+(j-center)**2)
          kernel[i,j]=np.exp(-(diff**2)/(2*sigma**2))
    return kernel/np.sum(kernel)


def conv2 (image,kernel):

	r,c= np.shape(image) 

	#number of rows and columns to be padded. It is made 5 to see the effect of padding else 1 is sufficient for 3x3 kernel
	pad_val = 5
	
	# creating an empty array to include the padded values
	img_pad = np.zeros((r+(2*pad_val),c+(2*pad_val)),dtype="float")
	p_r,p_c = np.shape(img_pad)

	# integrating image with the empty array
	img_pad[pad_val:p_r-pad_val,pad_val:p_c-pad_val] = image
	
	m,n = np.shape(img_pad)

	res_img = np.zeros((r,c),dtype="float") # empty matrix to have the output of convolution

	k_r,k_c = np.shape(kernel)
	h = int(math.floor(k_c/2))
	v = int(math.floor(k_r/2))

	# convolution of kernel with the image
	for i in np.arange(5,r+5):
		for j in np.arange(5,c+5):
		
			sub_img = img_pad[i-h:i+h+1,j-h:j+h+1]
			k = (sub_img * kernel).sum()

			res_img[i-5, j-5] = k
	res_img = rescale_intensity(res_img, in_range=(0,255))
	res_img = (res_img*255).astype("uint8")
	return res_img

#function to perform downsampling
def image_downsamp(image,scale,img=[]):
		r,c,ch= np.shape(image)

		gkern2d=get_gauss_kernel(3,2) #gaussian kernel of size 3x3 and sigma of 2

		ch1,ch2,ch3 = cv2.split(image)
		bl = conv2(ch1, gkern2d)
		gr = conv2(ch2, gkern2d)
		rd = conv2(ch3, gkern2d)
		image = cv2.merge((bl,gr,rd))

		#downsampling by taking alternate rows and columns
		for i in range(0,int(np.ceil(c*scale))):
			cloc= int(np.ceil(i/scale))
			image[:,i]= image[:,cloc]

		for j in range(0,int(np.ceil(r*scale))):
			rloc = int(np.ceil(j/scale))
			image[j,:]= image[rloc,:]

		img = image[0:int(np.ceil(r*scale)),0:int(np.ceil(c*scale))]

		return img

#function to perform upsampling using nearest neighbour interpolation
def image_upsamp(image,img=[]):
		r,c,ch = np.shape(image)

		img = np.repeat(image,2,axis=1)
		img = np.repeat(img,2,axis=0)
		gkern2d=get_gauss_kernel(3,2)

		#gaussian blur for upsampled image
		ch1,ch2,ch3 = cv2.split(img)
		bl = conv2(ch1, gkern2d)
		gr = conv2(ch2, gkern2d)
		rd = conv2(ch3, gkern2d)
		img = cv2.merge((bl,gr,rd))

		return img

#function to compute gaussian and laplacian pyramids for the input image
def ComputePyr(image, num_layers):

	gimg = []
	gimg.append(image)
	
	for i in range(0,num_layers):

		gimg.append(image_downsamp(gimg[i],0.5))

	for i in range(0,len(gimg)):
		cv2.imshow('gaussian',gimg[i])
		cv2.waitKey(5000)


	limg = []

	for j in range(0,num_layers-1):
		rj,cj,ch= np.shape(gimg[j])

		lapl = gimg[j]-(image_upsamp(gimg[j+1])[0:rj,0:cj])
		limg.append(lapl)


	for k in range(0,num_layers-1):
		limg[k] = rescale_intensity(limg[k], in_range=(0,255))
		limg[k] = (limg[k]*255).astype("int8")
		cv2.imshow('laplacian',limg[k])
		cv2.waitKey(5000)


	return [gimg,limg]




	
image = cv2.imread('image.png')#input image

r,c,ch = np.shape(image)

l = min(r,c)

#taking the number of layers from te user input
layers = int(input("Enter the number of layers required: "))

#computing if the user defined number of layers is valid
max_l = math.log(l,2)
if layers > int(max_l):
	print("Invalid number of layers!")
	layers = int(max_l)-1

[gpyr,lpyr] = ComputePyr(image,layers)





