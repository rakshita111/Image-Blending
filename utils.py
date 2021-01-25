#functions for blending of grayscale images

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

#Function to perform 2D convolution
def conv2 (image,kernel):

	image = image.astype("float32")
	r,c= np.shape(image)

	pad_val = 5
	
	# creating an empty array to include the padded values
	img_pad = np.zeros((r+(2*pad_val),c+(2*pad_val)),dtype="float")
	p_r,p_c = np.shape(img_pad)

	# integrating image with the empty array
	img_pad[pad_val:p_r-pad_val,pad_val:p_c-pad_val] = image
	
	m,n = np.shape(img_pad)

	res_img = np.zeros((r,c),dtype="float") # empty matrix for the output of convolution

	k_r,k_c = np.shape(kernel)
	h = int(math.floor(k_c/2))
	v = int(math.floor(k_r/2))

	# convolution of kernel with the image
	for i in np.arange(5,r+5):
		for j in np.arange(5,c+5):
		
			sub_img = img_pad[i-h:i+h+1,j-h:j+h+1]
			k = (sub_img * kernel).sum()

			res_img[i-5, j-5] = k

	return res_img.astype("float")

#function for downsampling
def image_downsamp(image,scale,imgtype,img=[]): #imgtype=1 for RGB, 0 for grayscale
		
		if imgtype == 0:
			r,c= np.shape(image)
			gkern2d=get_gauss_kernel(5,2)
			image = conv2(image, gkern2d)

		else:

			r,c,ch= np.shape(image)
			gkern2d=get_gauss_kernel(5,2)
			ch1,ch2,ch3 = cv2.split(image)
			bl = conv2(ch1, gkern2d)
			gr = conv2(ch2, gkern2d)
			rd = conv2(ch3, gkern2d)
			image = cv2.merge((bl,gr,rd))


		for i in range(0,int(np.ceil(c*scale))):
			cloc= int(np.ceil(i/scale))
			image[:,i]= image[:,cloc]

		for j in range(0,int(np.ceil(r*scale))):
			rloc = int(np.ceil(j/scale))
			image[j,:]= image[rloc,:]

		img = image[0:int(np.ceil(r*scale)),0:int(np.ceil(c*scale))]

		return img.astype("float32")

def image_upsamp(image,imgtype,img=[]): #function for upsampling

		if imgtype == 0:
			r,c= np.shape(image)

			img = np.repeat(image,2,axis=1)
			img = np.repeat(img,2,axis=0)
			gkern2d=get_gauss_kernel(5,2)
			img = conv2(img, gkern2d)

		else:
			r,c,ch = np.shape(image)


			img = np.repeat(image,2,axis=1)
			img = np.repeat(img,2,axis=0)
			gkern2d=get_gauss_kernel(5,2)


			ch1,ch2,ch3 = cv2.split(img)
			bl = conv2(ch1, gkern2d)
			gr = conv2(ch2, gkern2d)
			rd = conv2(ch3, gkern2d)
			img = cv2.merge((bl,gr,rd))


		return img.astype("float32")


#function to compute Gaussian and/or laplacian pyramids
def ComputePyr(image, num_layers, flag, imgtype): #flag for laplacian or gaussian

	gimg = []
	gimg.append(image)

	# gaussian pyramid
	for i in range(0,num_layers):
		gimg.append(image_downsamp(gimg[i],0.5,imgtype).astype("float32"))

	# laplacian pyramid
	if flag == 1:
		limg = []

		for j in range(0,num_layers):
			if imgtype == 0:
				rj,cj = np.shape(gimg[j])
			else:
				rj,cj = np.shape(gimg[j])
			lapl = np.subtract(gimg[j],(image_upsamp(gimg[j+1],imgtype))[0:rj,0:cj])
			limg.append((lapl).astype("float32"))
		return limg
	else:
		return gimg



