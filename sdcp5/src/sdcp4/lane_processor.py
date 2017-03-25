####################################################
## AUTHOR: James Beasley                          ##
## DATE: February 18, 2017                        ##
## UDACITY SDC: Project 4 (Advanced Lane Finding) ##
####################################################

#############
## IMPORTS ##
#############
import cv2
import numpy as np
import matplotlib.pyplot as plt
from threshold_processor import compute_hot_pixel_density_across_x_axis

#estimate the base location (index) of the lane lines using the hot (value of 1) pixel density across the bottom half of the image
def estimate_index_of_lane_line_base(image, export_debug_image=False):
    #set start position (y position...i.e., starting row number)
    offset = np.int(image.shape[0] / 2)
    #set window size (height...i.e., number of rows) that should be summed per x-axis column
    #this would normally be a fixed 'chunk', but to start, we're looking at the lower half of the image
    window_size = image.shape[0] - offset 
    #compute pixel peaks across the x-axis of the image
    hot_pixel_density_histogram = compute_hot_pixel_density_across_x_axis(image, offset, window_size)
    #locate the peak of the left and right halves of the histogram
    #these will be the starting point for the left and right lane lines
    #divide the vector in half (get midpoint)
    midpoint_index = np.int(hot_pixel_density_histogram.shape[0] / 2)
    #return the index of the largest value in the vector from 0 to (midpoint - 1) ...this should be the base of the left lane line
    left_lane_line_base_index = np.argmax(hot_pixel_density_histogram[:midpoint_index])
    #return the index of the largest value in the vector from (midpoint + 1) to (hot_pixel_density_histogram.shape[0] - 1) ...this should be the base of the right lane line
    #add the midpoint to the returned index to offset it correctly (since we're looking at the second half by itself, the argmax index returned will be wrong)
    right_lane_line_base_index = np.argmax(hot_pixel_density_histogram[midpoint_index:]) + midpoint_index
    #if true export a debug image
    if (export_debug_image):
        #plot result
        plt.plot(hot_pixel_density_histogram, color='b', linewidth=1)
        plt.xlabel('Pixel position', fontsize=14)
        plt.ylabel('Hot pixel density', fontsize=14)
        plt.savefig("output_images/stage3_hot_pixel_density_histogram_straight_lines1.jpg")
    #return estimated lane line locations
    return (left_lane_line_base_index, right_lane_line_base_index)

#compute the polynomial coefficients based on the supplied left and right lane line pixel coordinates
#each parameter contains the (x, y) coordinates estimated to be associated with the left and right lane
def compute_lane_line_coefficients(left_lane_line_pixel_coordinates, right_lane_line_pixel_coordinates):
    #fit (minimize squared error) a second order polynomial to the supplied points for each lane line
    #fitting for f(y) instead of f(x), as the lane lines in the warped image are near vertical and may have the same x value for more than one y value
    left_lane_line_coeff = np.polyfit(left_lane_line_pixel_coordinates[:, 0], left_lane_line_pixel_coordinates[:, 1], deg=2)
    right_lane_line_coeff = np.polyfit(right_lane_line_pixel_coordinates[:, 0], right_lane_line_pixel_coordinates[:, 1], deg=2)
    #return polynomial coefficients for the fitted left and right lane lines
    return (left_lane_line_coeff, right_lane_line_coeff)

#compute the offset of the vehicle in the lane in meters
def compute_vehicle_offset(image_size, left_lane_line_coeff, right_lane_line_coeff):
    standard_lane_width = 3.7 #standard width between the left and right lane line 
    #evaluate the x value of the fitted left and right lane polynomials at their base (i.e., at the height of image - 720 - remember image grows down) using the supplied coefficients
    left_lane_line_base = (left_lane_line_coeff[0] * (image_size[0] ** 2)) + (left_lane_line_coeff[1] * image_size[0]) + left_lane_line_coeff[2]
    right_lane_line_base = (right_lane_line_coeff[0] * (image_size[0] ** 2)) + (right_lane_line_coeff[1] * image_size[0]) + right_lane_line_coeff[2]
    #calculate the lane center (midpoint between base of calcualted left and right lane line)
    lane_center = np.mean([left_lane_line_base, right_lane_line_base])
    #calculate the image center
    image_center = np.int(image_size[1] / 2)
    #compute meters per pixel scaling factor (x-axis)
    meters_per_pixel = standard_lane_width / np.abs(left_lane_line_base - right_lane_line_base)
    #return calculated offset (in meters)
    return ((image_center - lane_center) * meters_per_pixel)

#compute the radius of curvature of the fitted lines in real world space (meters)
def compute_curvature_of_lane_lines(image_size, left_lane_line_fitted_poly, right_lane_line_fitted_poly):
    #define conversions in x and y from pixels space to meters
    y_meters_per_pixel = 30 / 720 #meters per pixel in y dimension
    x_meters_per_pixel = 3.7 / 700 #meters per pixel in x dimension
    #generate range of evenly spaced numbers over y interval (0 - 719) matching image height
    y_linespace = np.linspace(0, (image_size[0] - 1), image_size[0])
    #fit new polynomials (using polys fit to pixel space) to x, y in world space
    left_lane_line_coeff_rescaled = np.polyfit((y_linespace * y_meters_per_pixel), (left_lane_line_fitted_poly * x_meters_per_pixel), deg=2)
    right_lane_line_coeff_rescaled = np.polyfit((y_linespace * y_meters_per_pixel), (right_lane_line_fitted_poly * x_meters_per_pixel), deg=2)
    #calculate the new radii of curvature    
    radius_of_curvature_left = ((1 + (2 * left_lane_line_coeff_rescaled[0] * np.max(y_linespace) * y_meters_per_pixel + left_lane_line_coeff_rescaled[1]) ** 2) ** 1.5) / np.absolute(2 * left_lane_line_coeff_rescaled[0])
    radius_of_curvature_right = ((1 + (2 * right_lane_line_coeff_rescaled[0] * np.max(y_linespace) * y_meters_per_pixel + right_lane_line_coeff_rescaled[1]) ** 2) ** 1.5) / np.absolute(2 * right_lane_line_coeff_rescaled[0])
    #return curvature of left and right lane lines
    return (radius_of_curvature_left, radius_of_curvature_right)
    
#map out the lane line pixel locations using previously computed coefficients as a starting location to mount the search from in the supplied image 
def perform_educated_lane_line_pixel_search(image, prev_left_lane_line_coeff, prev_right_lane_line_coeff, prev_left_lane_line_fitted_poly=None, prev_right_lane_line_fitted_poly=None, return_debug_image=False):
    #return the [y, x] coordinates (i.e., row, col format) of all hot (value of 1) pixels in the binary image
    hot_pixel_coordinates = np.transpose(np.nonzero(image))
    #set width of the window +/- margin around left and right fitted polynomials
    window_margin = 100
    #if debug is set, the search windows are visualized on a returned debug image
    debug_image = None
    
    #previous left lane fitted polynomial (f(y) = A(y^2) + By + C)
    #these are used to filter x values, not to plot with, as the y vector dimensions won't make sense (its size is equal to the number of hot pixels in the current image, not the height of the image)
    left_lane_line_filter_poly = (prev_left_lane_line_coeff[0] * (hot_pixel_coordinates[:, 0] ** 2)) + (prev_left_lane_line_coeff[1] * hot_pixel_coordinates[:, 0]) + prev_left_lane_line_coeff[2]
    #previous right lane fitted polynomial (f(y) = A(y^2) + By + C)
    right_lane_line_filter_poly = (prev_right_lane_line_coeff[0] * (hot_pixel_coordinates[:, 0] ** 2)) + (prev_right_lane_line_coeff[1] * hot_pixel_coordinates[:, 0]) + prev_right_lane_line_coeff[2]
    
    #retrieve the [y, x] coordinates for all hot pixels located within the bounds of the current left and right windows (these bounds are a +/- margin from the previously fitted polynomial)
    #set selection conditions
    left_window_condition = ((hot_pixel_coordinates[:, 1] > (left_lane_line_filter_poly - window_margin)) & (hot_pixel_coordinates[:, 1] < (left_lane_line_filter_poly + window_margin)))  
    right_window_condition = ((hot_pixel_coordinates[:, 1] > (right_lane_line_filter_poly - window_margin)) & (hot_pixel_coordinates[:, 1] < (right_lane_line_filter_poly + window_margin)))
    #resulting coordinates chosen
    left_window_pixel_coordinates = hot_pixel_coordinates[left_window_condition]
    right_window_pixel_coordinates = hot_pixel_coordinates[right_window_condition]
    
    #if true return a debug image
    if (return_debug_image):
        #create an output image to draw on and visualize the result
        debug_image = np.dstack((image, image, image)) * 255
        #we'll draw the window representation on this image and overlay
        window_image = np.zeros_like(debug_image)
        #color all left lane pixels red
        #array is row (y), col (x)
        debug_image[left_window_pixel_coordinates[:, 0], left_window_pixel_coordinates[:, 1]] = [255, 0, 0]
        #color all right lane pixels blue
        #array is row (y), col (x)
        debug_image[right_window_pixel_coordinates[:, 0], right_window_pixel_coordinates[:, 1]] = [0, 0, 255]
        #generate range of evenly spaced numbers over y interval (0 - 719) matching image height
        y_linespace = np.linspace(0, (image.shape[0] - 1), image.shape[0])
        #recast the x and y points into usable format for fillPoly
        left_line_left_half_window = np.array([np.transpose(np.vstack([prev_left_lane_line_fitted_poly - window_margin, y_linespace]))])
        left_line_right_half_window = np.array([np.flipud(np.transpose(np.vstack([prev_left_lane_line_fitted_poly + window_margin, y_linespace])))])
        right_line_left_half_window = np.array([np.transpose(np.vstack([prev_right_lane_line_fitted_poly - window_margin, y_linespace]))])
        right_line_right_half_window = np.array([np.flipud(np.transpose(np.vstack([prev_right_lane_line_fitted_poly + window_margin, y_linespace])))])
        left_line_pts = np.hstack((left_line_left_half_window, left_line_right_half_window))
        right_line_pts = np.hstack((right_line_left_half_window, right_line_right_half_window))
        #draw the search channel (+/- margin around the previously fitted polynomial)
        cv2.fillPoly(window_image, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_image, np.int_([right_line_pts]), (0,255, 0))
        #combine highlighted pixel debug image with highlighted window channel image
        debug_image = cv2.addWeighted(debug_image, 1, window_image, 0.3, 0)
    
    #return coordinates chosen
    return (left_window_pixel_coordinates, right_window_pixel_coordinates, debug_image)

#map out the lane line pixel locations from scratch in the supplied image via windowed search 
def perform_blind_lane_line_pixel_search(image, return_debug_image=False):
    #set number of tracking windows (windows that move toward hot pixel density)
    num_windows = 9
    #set height of the windows
    window_height = np.int(image.shape[0] / num_windows)
    #set width of the windows +/- margin
    window_margin = 100
    #set minimum number of pixels needing to be found to re-center window at their mean location
    min_pixel_count_to_recenter = 50
    #lists that contain left and right lane pixel coordinates
    all_left_lane_pixel_coordinates = []
    all_right_lane_pixel_coordinates = []
    #if debug is set, the search windows are visualized on a returned debug image
    debug_image = None
   
    #return the [y, x] coordinates (i.e., row, col format) of all hot (value of 1) pixels in the binary image
    hot_pixel_coordinates = np.transpose(np.nonzero(image))
 
    #estimate base location of lane lines using the hot (value of 1) pixel density counts in the lower half of the image
    left_lane_line_base_index, right_lane_line_base_index = estimate_index_of_lane_line_base(image, export_debug_image=return_debug_image)
   
    #current x-axis index positions for the left and right lane search windows (to be updated as each window migrates position with the density of lane line pixels)
    cur_left_lane_line_x_index = left_lane_line_base_index
    cur_right_lane_line_x_index = right_lane_line_base_index
 
    #if true return a debug image
    if (return_debug_image):
        #create an output image to draw on and visualize the result
        debug_image = np.dstack((image, image, image)) * 255
 
    #enumerate each window, identifying and capturing hot pixels located within each
    for cur_window in range(0, num_windows):
        
        #set window bounds & location
        #window height dimensions (same for both left and right)
        window_y_low = image.shape[0] - ((cur_window + 1) * window_height)
        window_y_high = image.shape[0] - (cur_window * window_height)
        #left window width dimensions & location
        left_window_x_low = cur_left_lane_line_x_index - window_margin
        left_window_x_high = cur_left_lane_line_x_index + window_margin
        #right window width dimensions & location
        right_window_x_low = cur_right_lane_line_x_index - window_margin
        right_window_x_high = cur_right_lane_line_x_index + window_margin
   
        #if true return a debug image
        if (return_debug_image):
            #draw the windows on the debug image
            cv2.rectangle(debug_image, (left_window_x_low, window_y_low), (left_window_x_high, window_y_high), (0, 255, 0), 2)
            cv2.rectangle(debug_image, (right_window_x_low, window_y_low), (right_window_x_high, window_y_high), (0, 255, 0), 2)
    
        #retrieve the [y, x] coordinates for all hot pixels located within the bounds of the current left and right windows
        #selection condition
        left_window_condition = ((hot_pixel_coordinates[:, 1] >= left_window_x_low) & (hot_pixel_coordinates[:, 1] < left_window_x_high)) & ((hot_pixel_coordinates[:, 0] >= window_y_low) & (hot_pixel_coordinates[:, 0] < window_y_high))  
        right_window_condition = ((hot_pixel_coordinates[:, 1] >= right_window_x_low) & (hot_pixel_coordinates[:, 1] < right_window_x_high)) & ((hot_pixel_coordinates[:, 0] >= window_y_low) & (hot_pixel_coordinates[:, 0] < window_y_high))
        #resulting coordinates chosen
        left_window_pixel_coordinates = hot_pixel_coordinates[left_window_condition]
        right_window_pixel_coordinates = hot_pixel_coordinates[right_window_condition]
        
        #if greater than min_pixels_count_to_recenter were found, recenter next window on their mean position
        if (len(left_window_pixel_coordinates) > min_pixel_count_to_recenter):
            #strip out all x coordinates and compute their mean
            cur_left_lane_line_x_index = np.int(np.mean(left_window_pixel_coordinates[:, 1]))
        if (len(right_window_pixel_coordinates) > min_pixel_count_to_recenter): 
            #strip out all x coordinates and compute their mean      
            cur_right_lane_line_x_index = np.int(np.mean(right_window_pixel_coordinates[:, 1]))
            
        #append the pixels found in the left and right windows to the appropriate coordinate list
        all_left_lane_pixel_coordinates.append(left_window_pixel_coordinates)
        all_right_lane_pixel_coordinates.append(right_window_pixel_coordinates)
 
    #concatenate the left and right lane line pixel coordinates
    all_left_lane_pixel_coordinates = np.concatenate(all_left_lane_pixel_coordinates)
    all_right_lane_pixel_coordinates = np.concatenate(all_right_lane_pixel_coordinates)
    
    #if true return a debug image
    if (return_debug_image):
        #color all left lane pixels red
        #array is row (y), col (x)
        debug_image[all_left_lane_pixel_coordinates[:, 0], all_left_lane_pixel_coordinates[:, 1]] = [255, 0, 0]
        #color all right lane pixels blue
        #array is row (y), col (x)
        debug_image[all_right_lane_pixel_coordinates[:, 0], all_right_lane_pixel_coordinates[:, 1]] = [0, 0, 255]
        #fyi: since we color the line pixels last, the window rectangles will look like they're in back of the identified lines (in a layer below)
 
    #return all left and right lane line pixel coordinates and debug image
    return (all_left_lane_pixel_coordinates, all_right_lane_pixel_coordinates, debug_image)