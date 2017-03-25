#########################################################
## AUTHOR: James Beasley                               ##
## DATE: March 18, 2017                                ##
## UDACITY SDC: Project 5 (Vehicle Detection/Tracking) ##
#########################################################

#############
## IMPORTS ##
#############
import cv2
import numpy as np
from skimage.feature import hog

###############
## FUNCTIONS ##
###############

#perform histogram of oriented gradients (hog) operation on a single channel of the image and return features (either as vector or multidimensional array - based on feature vector boolean)
def perform_hog_feature_extraction(image_channel, orientation_bins, pixels_per_cell, cells_per_block, feature_vector=True, export_debug_image=False):
    #local vars
    debug_image = None
    #if we need to export a debug image that visualizes the gradients
    if (export_debug_image):
        #perform hog feature extraction and generate debug image 
        features, debug_image = hog(image_channel, orientation_bins, (pixels_per_cell, pixels_per_cell), (cells_per_block, cells_per_block), True, True, feature_vector)
        #convert from rgb to bgr (the format opencv likes)
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("output_images/hog_visualization_test6.jpg", debug_image)
    else:
        #perform hog feature extraction and generate debug image 
        features = hog(image_channel, orientation_bins, (pixels_per_cell, pixels_per_cell), (cells_per_block, cells_per_block), False, True, feature_vector)
    #return features   
    return features

#reduce image resolution (spatial binning) while still preserving relevant features
def perform_spatial_reduction(image, new_size):
    #resize and unroll into feature vector
    return np.ravel(cv2.resize(image, (new_size, new_size))) 

#compute the image's pixel intensity frequency distribution  
def compute_pixel_intensity_frequency_distribution(image, number_of_bins):
    #compute pixel intensity frequency distribution for each channel
    channel_0_fd = np.histogram(image[:, :, 0], number_of_bins)
    channel_1_fd = np.histogram(image[:, :, 1], number_of_bins)
    channel_2_fd = np.histogram(image[:, :, 2], number_of_bins)
    #concatenate the histograms into a single feature vector and return 
    return np.concatenate((channel_0_fd[0], channel_1_fd[0], channel_2_fd[0]))