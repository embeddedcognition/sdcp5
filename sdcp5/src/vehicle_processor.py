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
from model_training.feature_processor import *

###############
## FUNCTIONS ##
###############

#for each bounding box, increment the values of all pixels within it (apply heat to those pixels)  
def apply_heat_to_heatmap(heatmap, bounding_box_list):
    #enumerate the list of bounding boxes and increment the values of all pixels within them
    for cur_bounding_box in bounding_box_list:
        #add += 1 for all pixels inside each bounding box (assuming each takes the form [(x1, y1), (x2, y2)])
        heatmap[cur_bounding_box[0][1]:cur_bounding_box[1][1], cur_bounding_box[0][0]:cur_bounding_box[1][0]] += 1
    #return updated heatmap
    return heatmap

#apply a threshold to the heatmap, then return the result
def apply_threshold_to_heatmap(heatmap, threshold):
    #zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    #return thresholded map
    return heatmap

#draw bounding boxes on an image
def draw_bounding_boxes(image, bounding_box_list, line_color=(102, 0, 0), line_thickness=5):
    #make a copy of the supplied image to draw on
    draw_image = np.copy(image)
    #enumerate the bounding boxes and draw them on the image
    for cur_bounding_box in bounding_box_list:
        #draw the current bounding box on the image
        cv2.rectangle(draw_image, cur_bounding_box[0], cur_bounding_box[1], line_color, line_thickness)
    #return the image with the bounding boxes drawn on it
    return draw_image

#draw bounding boxes for supplied labeled objects
def draw_bounding_boxes_for_labeled_objects(image, labeled_objects):
    #make a copy of the supplied image to draw on
    draw_image = np.copy(image)
    #enumerate the count of labeled objects id's in the image
    for labeled_object_id in range(1, labeled_objects[1] + 1):
        #return the [y, x] coordinates (i.e., row, col format) of all pixel's whose values match the current labeled object id
        labeled_object_pixel_coordinates = np.transpose(np.nonzero(labeled_objects[0] == labeled_object_id))
        #compute the bounding box for this labeled object based on the min/max of x and y
        computed_bounding_box = ((np.min(labeled_object_pixel_coordinates[:, 1]), np.min(labeled_object_pixel_coordinates[:, 0])), (np.max(labeled_object_pixel_coordinates[:, 1]), np.max(labeled_object_pixel_coordinates[:, 0])))
        #draw the computed bounding box on the image
        cv2.rectangle(draw_image, computed_bounding_box[0], computed_bounding_box[1], (100, 0, 0), 5)
    #return the image with the computed bounding objects for the supplied labeled objects
    return draw_image

#perform vehicle search at multiple image scales (via list of provided scales)
#predict vehicle presence in a frame and return list of locations for all vehicles detected
def perform_vehicle_search(image, y_axis_start, y_axis_stop, scale_factor_list, support_vector_classifier, X_feature_scaler, spatial_reduction_size, pixel_intensity_fd_bins, hog_orientation_bins, hog_pixels_per_cell, hog_cells_per_block):
    #local vars
    positive_detection_window_coordinates = []  #list of positive detection window coordinates
    #search for vehicles at all supplied scales, storing the positive predictions
    for cur_scale_factor in scale_factor_list:
        #crop the input image (via y-axis) to our region of interest
        rgb_cropped_image = image[y_axis_start:y_axis_stop, :, :]
        #convert the cropped rgb image to the ycrcb color space
        ycrcb_cropped_image = cv2.cvtColor(rgb_cropped_image, cv2.COLOR_RGB2YCR_CB)
        #if the current scale factor is larger or smaller than the current scale (i.e., 64x64)  
        if (cur_scale_factor != 1):
            ycrcb_cropped_image_shape = ycrcb_cropped_image.shape
            #rescale the cropped image using the cur scale factor
            ycrcb_cropped_image = cv2.resize(ycrcb_cropped_image, (np.int(ycrcb_cropped_image_shape[1] / cur_scale_factor), np.int(ycrcb_cropped_image_shape[0] / cur_scale_factor)))
        #split channels
        y_channel = ycrcb_cropped_image[:, :, 0]
        cr_channel = ycrcb_cropped_image[:, :, 1]
        cb_channel = ycrcb_cropped_image[:, :, 2]
        #define blocks and steps as above
        nxblocks = (y_channel.shape[1] // hog_pixels_per_cell) - 1
        nyblocks = (y_channel.shape[0] // hog_pixels_per_cell) - 1 
        nfeat_per_block = hog_orientation_bins * (hog_cells_per_block ** 2)
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // hog_pixels_per_cell) - 1 
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        #compute the hog features for each channel of the entire image, returning a multidimensional array to sample from to get the features for each patch
        y_channel_hog_features = perform_hog_feature_extraction(y_channel, hog_orientation_bins, hog_pixels_per_cell, hog_cells_per_block, False, False)
        cr_channel_hog_features = perform_hog_feature_extraction(cr_channel, hog_orientation_bins, hog_pixels_per_cell, hog_cells_per_block, False, False)
        cb_channel_hog_features = perform_hog_feature_extraction(cb_channel, hog_orientation_bins, hog_pixels_per_cell, hog_cells_per_block, False, False)
    
        #travese the grid
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                xleft = xpos * hog_pixels_per_cell
                ytop = ypos * hog_pixels_per_cell
                
                # Extract HOG for this patch
                #given the discontinuities at the edges of the sub-images that will be present given this extraction
                #method (vs. running hog on a patch at a time) false positives will increase....the tradeoff is for speed
                hog_feat1 = np.ravel(y_channel_hog_features[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window])
                hog_feat2 = np.ravel(cr_channel_hog_features[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]) 
                hog_feat3 = np.ravel(cb_channel_hog_features[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]) 
                #horisontally stack the features into a vector
                raw_pixel_intensity_gradient_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                #extract image patch
                cur_image_patch = cv2.resize(ycrcb_cropped_image[ytop:ytop+window, xleft:xleft+window], (64, 64))
          
                ## EXTRACT RAW PIXEL INTENSITY FEATURES (TARGETING COLOR & SHAPE) ##
                #reduce resolution while still preserving relevant features
                #raw pixel intensities (even when down-sampled) reveal color and shape characteristics
                raw_pixel_intensity_features = perform_spatial_reduction(cur_image_patch, spatial_reduction_size)
                #compute pixel intensity frequency distribution (all three channels will be computed separately and concatenated
                #computing a frequency distribution of the raw pixel intensities reveals only color characteristics
                raw_pixel_intensity_fd_features = compute_pixel_intensity_frequency_distribution(cur_image_patch, pixel_intensity_fd_bins)
                #scale features
                scaled_features = X_feature_scaler.transform(np.hstack((raw_pixel_intensity_features, raw_pixel_intensity_fd_features, raw_pixel_intensity_gradient_features)).reshape(1, -1))    
                #make prediction
                prediction = support_vector_classifier.predict(scaled_features)
            
                #if we have a positive prediction (i.e., vehicle) add the window coordinates to the list
                if ((prediction == 1) and (support_vector_classifier.decision_function(scaled_features) > 1.1)):
                    xbox_left = np.int(xleft * cur_scale_factor)
                    ytop_draw = np.int(ytop * cur_scale_factor)
                    win_draw = np.int(window * cur_scale_factor)
                    #add this window's coords to our list
                    positive_detection_window_coordinates.append([(xbox_left, ytop_draw + y_axis_start), (xbox_left + win_draw, ytop_draw + win_draw + y_axis_start)]) 
    #return     
    return positive_detection_window_coordinates