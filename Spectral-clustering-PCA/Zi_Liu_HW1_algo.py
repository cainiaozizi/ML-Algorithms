#!/usr/bin/env python
# coding: utf-8

# ## ISYE 6740 - Computational Data Analysis - Homework 1
# ## Zi Liu

#import all required packages

import time
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow, subplots


# #### The following section defines all the functions relate to image processing:

# In[2]:


#Define functions for reading and displaying the images

def read_image(path):
    """
    ptype: image path
    rtype: 3D image array
    """
    img = Image.open(path)
    img_arr = np.array(img, dtype='int64')
    img.close()
    return img_arr

def show_image(img_array):
    """
    ptype: 3D image array
    """
    img_array = img_array.astype(dtype='uint8')
    img = Image.fromarray(img_array, 'RGB')

    return np.asarray(img)

#use to stack the sequence of input arrays vertically to make a single array.    
def reshape_image(img_array):
    """
    ptype: 3D image array
    rtype: 2D image array
    """
    a = np.vstack(img_array)
    
    return a

def comepress_image(algorithm,k,image_array):
    '''
    algorithm: the returning results from k_means function
    k: number of clusters
    image_array: original image before reshape
    '''
    #preparation work
    kmean_centers,kmean_labels = algorithm
    row,column,dim = image_array.shape
    centers_dict = {}
    
    for i in range(k):
        centers_dict[i] = kmean_centers[i]
        
    img_compress = np.array([centers_dict[i] for i in kmean_labels])
    img_disp = np.reshape(img_compress,(row,column,dim),order = "C")
    
    return show_image(img_disp)
    


# #### The K-means algorithm is defined in the following section. It is built based on several sub-fucntions:
# * init_centers()
# * assign_cluster()
# * update_centers()
# * check_converage()
# 


#k-means algorithm

#Define all the functions here.

def init_centers(data, k):
    num_samples, dim = data.shape
    centroids = np.zeros((k, dim))
    
    for i in range(k):
        index = int(np.random.uniform(0, num_samples))
        centroids[i, :] = data[index, :]
        
    return centroids

    
#Assign the cluster based on the shortest Euclidean distance 
def assign_cluster(centroids,data):
    distance = (((pow(data[:,:,None] - centroids.T[None,:,:], 2)))**0.5).sum(axis = 1) #Euclidean distance   
    labels = np.argmin(distance,axis = 1)
    
    return labels

def update_centers(data,labels):
    num_of_pts,dim = data.shape #dim == 3 for sure
    num_of_labels = len(np.unique(labels))
    
    if num_of_pts != len(labels):
        print('for self check:num_of_pts != len(labels)!')
    
    c = np.empty((num_of_labels,dim))
    
    for i in range(num_of_labels):
        pts_in_cluster = data[labels ==i,:]
        c[i,:] = np.mean(pts_in_cluster,axis = 0)
    
    return c

def check_converage(b4_center,current_centers):
    if set([tuple(x) for x in b4_center]) == set([tuple(x) for x in current_centers]):
        return True
    
    return False

def k_means(data,k,max_iteration= np.inf,seedNum=123):
    #preparation work
    np.random.seed(seedNum)
    num_samples = data.shape[0]
    
    labels = np.empty(num_samples) #create an empty array to store labels
    coveraged_flag = False
    iteration = 1
    
    #step 1: initiate centriods
    centers = init_centers(data,k)
    
    #step 2: loop
    while not coveraged_flag and iteration <= max_iteration:
        old_centers = centers
        labels = assign_cluster(centers,data)
        centers = update_centers(data,labels)
        
        if check_converage(old_centers,centers) == True:
            coveraged_flag = True
            
            print("number of iterations:",iteration)
            return centers,labels
        
        iteration += 1
        
    return centers,labels
