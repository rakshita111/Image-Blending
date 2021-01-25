# BLENDING OF TWO GRAYSCALE IMAGES

import cv2 
import argparse 
import numpy as np
from skimage.exposure import rescale_intensity
from utils import get_gauss_kernel, conv2, image_downsamp, image_upsamp, ComputePyr

#function to spread the itensity values of the output image linearly
def spread_linear(img,upper_lim):
    s = np.copy(img)
    maxm = img.max()
    minm = img.min()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            s[i,j] = ((upper_lim - 1)/(maxm - minm)) * (img[i,j] - minm)

    return s.astype('uint8')

ref_point = [] 

def shape_selection(event, x, y, flags, param):
    global ref_point 
  
	#recording coordinate values for click of left button 
    if event == cv2.EVENT_LBUTTONDOWN: 
        ref_point = [(x, y)] 
  
    # recording coordinate values for the release of mouse button 
    elif event == cv2.EVENT_LBUTTONUP:  
        ref_point.append((x, y)) 
  
        # drawing rectangle for selected ROI 
        cv2.rectangle(img, ref_point[0], ref_point[1], (255, 255, 255), 2) 
        cv2.imshow("image", img) 
  
  
# parsing the arguments 
ap = argparse.ArgumentParser() 

args = vars(ap.parse_args()) 

#reading the foreground image
img =  cv2.imread('foreground_image.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
fg = cv2.imread('foreground_image.jpg')
fg = cv2.cvtColor(fg,cv2.COLOR_BGR2GRAY)
r,c = np.shape(fg)


cv2.namedWindow("image") 
cv2.setMouseCallback("image", shape_selection) 
  
  
#break from the loop when 'c' key is pressed
while True:  
    cv2.imshow("image", img) 
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"): 
        break

#creating mask for the foreground image
mask = np.zeros_like(fg,dtype = "uint8")

mask[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]] = 255


#displaying the mask
cv2.imshow("crop_img",mask) 
cv2.waitKey(5000)


im2 =  cv2.imread('background_image.jpg')
bg = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

#diaplaying the background image
cv2.imshow("background image",bg) 
cv2.waitKey(200)


#computing Laplacian and Gaussian pyramids for the input images and the mask
lap1 =  ComputePyr(fg.astype("float"),3,1,0)
gau1 = ComputePyr(mask,3,0,0)
lap2 =  ComputePyr(bg.astype("float"),3,1,0)

#converting the mask values to float
for i in range(len(gau1)):
	gau1[i] = (gau1[i]/255).astype("float")


#generating the blended pyramids
term1 = []
com1 = []

term2 = []
com2 = []

blend = []
k = 0
for i in range (len(lap1)-1,-1,-1):
    com1 = np.multiply(lap1[i],gau1[i])
    term1.append(com1)

    com2 = np.multiply(lap2[i],(1.0-gau1[i]))
    term2.append(com2)
 

    blend.append(term1[k]+term2[k])
    k += 1

#collapsing the blended pyramids to obtain the blended image
res = []
res.append(blend[0])

gkern2d = get_gauss_kernel(5,2)
for i in range(0,len(blend)-1):

    tmp1 = image_upsamp(res[i],0)	
    tmp1= conv2(tmp1, gkern2d)
    tmp2 = blend[i+1]

    total = np.add(tmp1 , tmp2)

    res.append(total)
    output = total


#linearly spreading the intensity values to get the final image
res_img = spread_linear(output,255)

#diaplaying the final blended image
cv2.imshow("blended image",res_img)
cv2.waitKey(0)
