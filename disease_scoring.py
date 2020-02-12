
# coding: utf-8

# In[122]:


### Authors: Aditya Jain and Taneea S Agrawaal (IIIT-Delhi) #####
### Topic: Mosaicing of Drone Imagery ###
### Start Date: 10th March, 2018 ###

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
import os
# import math
# Crop target object
def crop_object(img, img_thrs):
    """
    Parameters
    ----------
    img : image (RGB)
        image
           
    Returns
    -------
    out
        image 2D
    """
    #im2, contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_TREE, 
     #                                       cv2.CHAIN_APPROX_SIMPLE)
    #sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    x, y, w, h = cv2.boundingRect(img_thrs)
# Getting ROI
    roi = img[y:y + h, x:x + w,:]
    return(roi)


# Segment object
def filter_frame(img_copy, min, max):
    """
    Parameters
    ----------
    img : image (RGB)
        image
           
    Returns
    -------
    out
        image 2D
    """
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, min, max, cv2.THRESH_BINARY)
    
    return(thresholded)
    
#Pre-process whole image   
def postproc_custom1(binay_img, bx):
    """
    Parameters
    ----------
    img : image (RGB)
        image
           
    Returns
    -------
    out
        image 2D
    """
    kernel = np.ones((bx, bx),np.uint8)
    erosion = cv2.erode(binay_img, kernel, iterations = 1)
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    im_floodfill = closing.copy()
    # Mask used to flood filling.
    h, w = closing.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    #im_out = closing | im_floodfill_inv
    return(im_floodfill_inv)




# Convert image to double
def im2double(img):
    """
    Parameters
    ----------
    img : image (RGB)
        image
           
    Returns
    -------
    out
        image 2D
    """  
    info = np.iinfo(img.dtype) # Get the data type of the input image
    out = img.astype(np.float) / info.max
    return out


def ccGrayWorld(img):
  
    [row, col] = img.shape[0:2]
    im2d = img.reshape((row*col, 3))
    im2d = im2double(im2d)
  # #Grey World
  # illuminant corrected image
    imGW = im2double(img)
    c=0; imGW[:,:,c] = imGW[:,:,c] / cv2.mean(im2d[0:im2d.shape[0], c])[0]
    c=1; imGW[:,:,c] = imGW[:,:,c] / cv2.mean(im2d[0:im2d.shape[0], c])[0]
    c=2; imGW[:,:,c] = imGW[:,:,c] / cv2.mean(im2d[0:im2d.shape[0], c])[0]
    u = np.uint8(np.round(imGW*255))
      
    return(u)

  
def ccMaxRGB(img):
  [row, col] = img.shape[0:2]
  im2d = img.reshape((row*col, 3))
  im2d = im2double(im2d)
  #MaxRGB
  LMaxRGB = LMaxRGB = list()
  LMaxRGB.append(np.max(im2d[0:im2d.shape[0],0]))
  LMaxRGB.append(np.max(im2d[0:im2d.shape[0],1]))
  LMaxRGB.append(np.max(im2d[0:im2d.shape[0],2]))
 
  #% illuminant corrected image
  imMaxRGB = im2double(img)
  c=0; imMaxRGB[:,:,c] = imMaxRGB[:,:,c] / LMaxRGB[c];
  c=1; imMaxRGB[:,:,c] = imMaxRGB[:,:,c] / LMaxRGB[c];
  c=2; imMaxRGB[:,:,c] = imMaxRGB[:,:,c] / LMaxRGB[c];
  
  
  return(u)


# This function search for all images in a given directory
def search_images(path, kw):
    """
    Parameters
    ----------
    path : string
        Path to images
    kw : str
        Image type 
       
    Returns
    -------
    list
        list of image filenames
    """
    file_list = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if kw in file:
                file_list.append(os.path.join(r, file))
         
    return(file_list)
    
    
 # This function extract features from images
def get_features(f):
    """
    Parameters
    ----------
    f : string
        image filename
   
       
    Returns
    -------
    res_imc : dict
        dictionay with image features
    
        
    """
   
    #img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    
    img = cv2.imread(f)
    hsvImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    plot_image(hsvImg)
    hsvImg[...,2] = hsvImg[...,2]*0.6

    plt.subplot(111), plt.imshow(cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB))
    plt.title('brightened image'), plt.xticks([]), plt.yticks([])
    plt.show()
    #equ = cv2.equalizeHist(img)
    #res = np.hstack((img, equ))
    #cv2.imwrite('res.png',res)
        #plt.imshow(img)
    crop_img = img[500:2000, 1000:3900]
    #print(f, np.mean(crop_img.flatten()))
    plot_image(crop_img)
    thres_obj = filter_frame(img, 250, 255)
        # post-process segmented image
    thres_obj = postproc_custom1(thres_obj)
        # Crop colour card
    img_cropt = crop_object(img, thres_obj)
        
        # Filter boxes in color card
    thres_boxes = filter_frame(img_cropt, 230, 255)
    plot_image(thres_boxes)
        # Select boxes
    thres_boxes[np.where(thres_boxes == 0)] = 1
    thres_boxes[np.where(thres_boxes == 255)] = 0
    kernel = np.ones((25, 25),np.uint8)
    thres_boxes = cv2.erode(thres_boxes, kernel, iterations = 1)
    
    im_floodfill = thres_boxes.copy()
    # Mask used to flood filling.
    h, w = thres_boxes.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    thres_boxes = cv2.bitwise_not(im_floodfill)
       
    #plot_image(thres_boxes)
        
    im2, contours, hierarchy = cv2.findContours(thres_boxes, cv2.RETR_TREE, 
                                                    cv2.CHAIN_APPROX_SIMPLE)
        
    box_dic = dict()
    for i in range(0, len(contours)) :
        roi = crop_object(img_cropt, contours[i])
        #plot_image(roi)
        box_dic[i] = np.mean(roi.flatten()), cv2.contourArea(contours[i])
            
    #plot_image(img_cropt)
       
             
    return(np.mean(crop_img.flatten()), box_dic)
          

# Get pixel value      
def impixel(img):
    """
     Parameters
    ----------
    img : image (RGB)
        image
           
    Returns
    -------
    out
        image 2D
    """
    
    #scale_width = 640 / img.shape[1]
    #scale_height = 480 / img.shape[0]
    #scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * 1)
    window_height = int(img.shape[0] * 1)
    #
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    cv2.imshow('image', img)
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows


#right-click event value 
def click_event(event, x, y, flags, param):
   
    if event == cv2.EVENT_LBUTTONDOWN:
        a = img.shape
        d = list()
        #print(a[2])
        for i in range(1, a[1]):
            d.append(img[y,x])
        
        red = img[y,x]
        #green = img[y,x,1]
        #blue = img[y,x,2]
       
        print(red)
        



# Plot image
def plot_image(img):
    """
    Parameters
    ----------
    img : image (RGB)
        image
           
    """
    plt.figure()
    plt.axis("on")
    plt.imshow(img)
    plt.show()




def get_features(fname, minp, maxp):
    """
    Parameters
    ----------
    f : string
        image filename
   
       
    Returns
    -------
    res_imc : dict
        dictionay with image features
    
        
    """
    
    image_raw = cv2.imread(fname)
   
    height, width = image_raw.shape[:2]
    testImage = cv2.resize(image_raw, (round(width / 4), round(height / 4)))
    plot_image(image_raw)
    # conver to gray
    gray = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    
    # Segment clusters    
    _, thresholded = cv2.threshold(gray, minp, maxp, cv2.THRESH_BINARY)
    
    plot_image(thresholded)
    nb_comp,output,sizes,centroids=cv2.connectedComponentsWithStats(thresholded,connectivity=4)
    print(nb_comp)
    #return(np.mean(crop_img.flatten()), box_dic)
          

def extractComponent(image,label_image,label):
    
    component = np.zeros(image.shape,np.uint8)
    #component[label_image==label]=image[label_image==label]
    component[label_image==label] = 1
    print(component)
    return component


def kmeans(image, segments):
    #Preprocessing step
    #segments = 5
    image=cv2.GaussianBlur(image,(7,7),0)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vectorized=image.reshape(-1,3)
    vectorized=np.float32(vectorized)
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(vectorized, segments,None, 
                                criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))
    
    
    return label.reshape((image.shape[0],image.shape[1])), segmented_image.astype(np.uint8), center,vectorized


def plot_centroids(vectorized, center) :

    y = {}
    for i in range(0, center.shape[1]) :
        y[i] = vectorized[label.ravel() == i]  
    # Plot the data
        plt.scatter(y[i][:,0], y[i][:,1])
        #plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
        plt.scatter(center[:,0], center[:,1], s = 80,c = 'y', marker = 's')

    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.show()




file_list = os.path.abspath("c:/anyela/repo/DiseaseScoring/*.JPG")


images = sorted(glob.glob(file_list))

for name in images[1]:
    image_raw = cv2.imread(name)
    #image_raw_gs = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
    #Z = image_raw_gs.reshape((-1,3))
    # convert to np.float32
    #Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #K = 4
    #ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    #center = np.uint8(center)
    #res = center[label.flatten()]
    #res2 = res.reshape((image_raw_gs.shape))
    
    #plot_image(res2)


k = 3
label,result, center, vectorized = kmeans(image_raw, k)
plot_centroids(vectorized, center)


for i in range(0,k) :
    print(i)
    extracted = extractComponent(image_raw, label,i)
    extracted = extracted[:,:,1]
    plot_image(extracted)
    #plt.imshow(extracted, cmap=plt.cm.gray)
     
extracted = postproc_custom1(extracted, 5)
nb_comp,output,sizes,centroids = cv2.connectedComponentsWithStats(extracted, connectivity=4)

extracted = extractComponent(image_raw, label,1)
extracted = extracted[:,:,1]
extracted = postproc_custom1(extracted, 5)
plt.imshow(extracted)