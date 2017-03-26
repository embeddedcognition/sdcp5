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
import pickle
from scipy.ndimage.measurements import label
from sdcp4.calibration_processor import perform_undistort
from sdcp4.perspective_processor import perform_perspective_transform
from sdcp4.threshold_processor import perform_thresholding
from sdcp4.lane_processor import *
from vehicle_processor import *

#test the pipeline components and produce outputs in the output_images directory
#perspective_transform_components[0] is warp_perspective_matrix, perspective_transform_components[1] is unwarp_perspective_matrix
def execute_test_pipeline(calibration_components, perspective_transform_components, src_vertices):
    
    #############################
    ## TEST CAMERA CALIBRATION ##
    #############################
    
    #test camera calibration by undistorting a test road image
    #load image
    bgr_test_road_image = cv2.imread("test_images/test6.jpg")
    #convert from bgr to rgb
    test_road_image = cv2.cvtColor(bgr_test_road_image, cv2.COLOR_BGR2RGB)

    #undistort image - this undistorted image will be used to demonstrate the production_pipeline along the way (all outputs will be placed in 'output_images' folder)
    undistorted_test_road_image = perform_undistort(test_road_image, calibration_components)

    ################################
    ## TEST PERSPECTIVE TRANSFORM ##
    ################################
    
    #transform perspective (warp) - this will squish the depth of field in the source mapping into the height of the image, 
    #which will make the upper 3/4ths blurry, need to adjust dest_upper* y-values to negative to stretch it out and clear the transformed image up
    #we won't do that as we'll lose right dashes in the 720 pix height of the image frame 
    warped_undistorted_test_road_image = perform_perspective_transform(undistorted_test_road_image, perspective_transform_components[0])

    ####################################
    ## TEST COLOR/GRADIENT THRESHOLD  ##
    ####################################

    #apply thresholding to warped image and produce a binary result
    thresholded_warped_undistorted_test_road_image = perform_thresholding(warped_undistorted_test_road_image)
    
    #########################
    ## TEST LANE DETECTION ##
    #########################
    
    ## BLIND SEARCH ##
    
    #map out the left and right lane line pixel coordinates via windowed search
    left_lane_pixel_coordinates, right_lane_pixel_coordinates, _ = perform_blind_lane_line_pixel_search(thresholded_warped_undistorted_test_road_image, return_debug_image=False)
    
    #compute the polynomial coefficients for each lane line using the x and y pixel locations from the mapping function
    #we're fitting (computing coefficients of) a second order polynomial: f(y) = A(y^2) + By + C
    #we're fitting for f(y) rather than f(x), as the lane lines in the warped image are near vertical and may have the same x value for more than one y value 
    left_lane_line_coeff, right_lane_line_coeff = compute_lane_line_coefficients(left_lane_pixel_coordinates, right_lane_pixel_coordinates)
    
    #generate range of evenly spaced numbers over y interval (0 - 719) matching image height
    y_linespace = np.linspace(0, (thresholded_warped_undistorted_test_road_image.shape[0] - 1), thresholded_warped_undistorted_test_road_image.shape[0])
    
    #left lane fitted polynomial (f(y) = A(y^2) + By + C)
    left_lane_line_fitted_poly = (left_lane_line_coeff[0] * (y_linespace ** 2)) + (left_lane_line_coeff[1] * y_linespace) + left_lane_line_coeff[2]
    #right lane fitted polynomial (f(y) = A(y^2) + By + C)
    right_lane_line_fitted_poly = (right_lane_line_coeff[0] * (y_linespace ** 2)) + (right_lane_line_coeff[1] * y_linespace) + right_lane_line_coeff[2]

    #draw the fitted polynomials on the debug_image and export
    #recast the x and y points into usable format for polylines and fillPoly
    pts_left = np.array([np.transpose(np.vstack([left_lane_line_fitted_poly, y_linespace]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane_line_fitted_poly, y_linespace])))])
    
    ## compute left and right lane curvature ##
    left_curvature, right_curvature = compute_curvature_of_lane_lines(thresholded_warped_undistorted_test_road_image.shape, left_lane_line_fitted_poly, right_lane_line_fitted_poly)
    
    ## compute vehicle offset from center ##
    vehicle_offset = compute_vehicle_offset(thresholded_warped_undistorted_test_road_image.shape, left_lane_line_coeff, right_lane_line_coeff)
    
    #####################################
    ## TEST PROJECTION BACK ONTO ROAD  ##
    #####################################
    
    #create an image to draw the lines on
    warped_lane = np.zeros_like(warped_undistorted_test_road_image).astype(np.uint8)

    #draw the lane onto the warped blank image
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(warped_lane, np.int_([pts]), (152, 251, 152))
    
    #draw fitted lines on image
    cv2.polylines(warped_lane, np.int_([pts_left]), False, color=(189,183,107), thickness=20, lineType=cv2.LINE_AA)
    cv2.polylines(warped_lane, np.int_([pts_right]), False, color=(189,183,107), thickness=20, lineType=cv2.LINE_AA)

    #transform perspective back to original (unwarp)
    warped_to_original_perspective = perform_perspective_transform(warped_lane, perspective_transform_components[1])

    #combine (weight) the result with the original image
    projected_lane = cv2.addWeighted(undistorted_test_road_image, 1, warped_to_original_perspective, 0.3, 0)
    
    #add tracking text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(projected_lane, 'Lane curvature: {0:.2f} meters'.format(np.mean([left_curvature, right_curvature])), (20, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(projected_lane, 'Vehicle offset: {0:.2f} meters'.format(vehicle_offset), (20, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    ######################################
    ## TEST VEHICLE DETECTION/TRACKING  ##
    ######################################
    
    #hyperparameters
    spatial_reduction_size = 32   #reduce the training images from 64x64 to 32x32 resolution (smaller feature vector but still retains useful shape and color information)
    pixel_intensity_fd_bins = 64  #number of bins to use to compute raw pixel intensity frequency distribution
    hog_orientation_bins =  9     #number of orientation bins to use in hog feature extraction
    hog_pixels_per_cell = 8       #number of pixels per cell to use in hog feature extraction
    hog_cells_per_block = 2       #number of cells per block to use in hog feature extraction
    scale_factor_list = [2.0, 1.5, 1.2, 1] #window scales
    y_axis_start = 400 #start y-axis crop
    y_axis_stop = 656  #end y-axis crop

    dist_pickle = pickle.load( open("model_training/pickled_objects/trained_model.p", "rb" ) )
    support_vector_classifier = dist_pickle["model"]
    X_feature_scaler = dist_pickle["scaler"]
    
    #for the current frame, detect vehicles at each scale, 
    #returning the list of coordinates (p1 and p2) of the window that signaled a positive prediction
    positive_detection_window_coordinates = perform_vehicle_search(undistorted_test_road_image, y_axis_start, y_axis_stop, scale_factor_list, support_vector_classifier, X_feature_scaler, spatial_reduction_size, pixel_intensity_fd_bins, hog_orientation_bins, hog_pixels_per_cell, hog_cells_per_block)

    #create a heat map
    heatmap = np.zeros_like(undistorted_test_road_image[:,:,0]).astype(np.float)
    
    #apply heat to all pixels within the set of detected windows
    heatmap = apply_heat_to_heatmap(heatmap, positive_detection_window_coordinates)
    
    #apply threshold to heatmap to help remove false positives
    #heatmap = apply_threshold_to_heatmap(heatmap, 3)
    
    #compute final bounding boxes from heatmap
    labeled_objects = label(heatmap)
    
    #draw final bounding boxes on projected lane image
    projected_lane = draw_bounding_boxes_for_labeled_objects(projected_lane, labeled_objects)
    
    #save image
    #convert from rgb to bgr (the format opencv likes)
    projected_lane = cv2.cvtColor(projected_lane, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_images/projected_lane_and_detected_vehices_test6.jpg", projected_lane)