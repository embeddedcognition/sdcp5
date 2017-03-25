####################################################
## AUTHOR: James Beasley                          ##
## DATE: February 18, 2017                        ##
## UDACITY SDC: Project 4 (Advanced Lane Finding) ##
####################################################

#############
## IMPORTS ##
#############
import numpy as np
import cv2

#compute the density of hot (value of 1) pixels across the x-axis within a specified y-axis chunk/window 
#remember, (0, 0) of an image is the top left corner of that image
def compute_hot_pixel_density_across_x_axis(image, offset, window_size):
    #sum each column across the x-axis for the particular window in question (offset = y (row) start position, window_size = # of rows in each column to sum)
    #this will sum from offset to ((offset + window_size) - 1)
    return np.sum(image[offset:offset+window_size, :], axis=0)

#apply gaussian blur to an image
def apply_gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

#apply finite difference filter (Sobel) to an image
def apply_gradient_filter(image, orient='x', sobel_kernel=3, threshold=(0, 255)):
    #take the derivative in x or y given orient = 'x' or 'y'
    if (orient == 'x'):
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    #take the absolute value of the derivative/gradient
    abs_sobel = np.absolute(sobel)
    #scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8((255 * abs_sobel) / np.max(abs_sobel))
    #create a mask of 1's where the scaled gradient magnitude is > min and <= max
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel > threshold[0]) & (scaled_sobel <= threshold[1])] = 1
    #return result
    return binary

#apply color thresholding to the h, l, & s channels to enhance yellow and white lines
def apply_hls_channel_color_thresholding(h, l, s):
    #hue (represents color independent of any change in brightness)
    h_threshold = (0, 50)
    #treat the original as immutable
    filtered_h = h.copy()
    #values outside of the threshold range are set to zero
    filtered_h[(h < h_threshold[0]) | (h > h_threshold[1])] = 0
    #lightness (brightness)
    l_threshold = (140, 255)
    #treat the original as immutable
    filtered_l = s.copy()
    #values outside of the threshold range are set to zero
    filtered_l[(l < l_threshold[0]) | (l > l_threshold[1])] = 0
    #saturation (measurement of colorfulness)
    s_threshold = (140, 255)
    #treat the original as immutable
    filtered_s = s.copy()
    #values outside of the threshold range are set to zero
    filtered_s[(s < s_threshold[0]) | (s > s_threshold[1])] = 0
    #recombine filtered hls channels
    filtered_hls = np.dstack((filtered_h, filtered_l, filtered_s))
    #convert back to rgb color-space for display
    filtered_rgb = cv2.cvtColor(filtered_hls, cv2.COLOR_HLS2RGB)     
    #convert to binary
    binary = cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2GRAY)
    binary[binary < 128] = 0    #black
    binary[binary >= 128] = 1   #white
    #return 
    return binary

#apply gradient thresholding to the hls 'lightness' channel (l)
def apply_l_channel_gradient_thresholding(l):
    #smooth channel (blurring first allows us to set a higher 'low' threshold - e.g., less noise)
    l_blurred = apply_gaussian_blur(l, 45)
    #apply finite difference filter (Sobel - across x-axis)
    return apply_gradient_filter(l_blurred, orient='x', threshold=(45, 255))

#apply gradient & value thresholding to the hls 'saturation' channel (s)
def apply_s_channel_gradient_and_value_thresholding(s):
    #apply finite difference filter (Sobel - across x-axis)
    s_sobel_x = apply_gradient_filter(s, orient='x', threshold=(15, 255))
    #apply value thresholding to raw channel as well
    s_filter = np.zeros_like(s)
    s_filter[(s >= 150) & (s <= 255)] = 1
    #'or' the two and return
    return cv2.bitwise_or(s_sobel_x, s_filter)

#applies a color/gradient threshold process to an undistorted and warped (perpective transformed - bird's eye view) rgb image
def perform_thresholding(image):
    #convert to hls color space 
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    #extract all channels
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    #perform hls-channel color thresholding and return a binary image
    hls_binary = apply_hls_channel_color_thresholding(h, l, s)
    #perform l-channel gradient thresholding and return binary image
    l_binary = apply_l_channel_gradient_thresholding(l)
    #perform s-channel gradient and value thresholding and return binary image
    s_binary = apply_s_channel_gradient_and_value_thresholding(s)
    #compute the hot pixel density score for the s_binary image (build resiliency against degraded image due to difficult frame)
    #get the count of non-zero pixels in the image (i.e., how many 1's are there) - just counting the number of y-coordinates returned 
    #(could have also counted just the x-coordinates)
    s_binary_hot_pixel_count = len((s_binary.nonzero())[0])
    #density score is the number of positive (hot) pixels in the image divided by the total number of pixels in the image
    s_binary_density_score = s_binary_hot_pixel_count / (s_binary.shape[0] * s_binary.shape[1])
    #combine the 'hls' and 'l' binary images
    final_binary_image = cv2.bitwise_or(hls_binary, l_binary)
    #if the s_binary image has a sufficiently low hot pixel density we're more confident in the fidelity of the line definition 
    #a good density score (s_binary image with solid left and dashed right identified) will be ~0.03
    #a terrible density score (s_binary image with a lot of clouding/blotching) will be ~0.40
    #0.15 is an arbitrary threshold that gives a lot of headroom for variation and noise
    if (s_binary_density_score < 0.15):
        #combine the or'ed 'hls' and 'l' images with the s_binary image
        final_binary_image = cv2.bitwise_and(final_binary_image, s_binary)
    #return
    return final_binary_image