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
from collections import deque
from moviepy.editor import VideoFileClip
from sdcp4.calibration_processor import perform_undistort
from sdcp4.perspective_processor import perform_perspective_transform
from sdcp4.threshold_processor import perform_thresholding
from sdcp4.lane_processor import perform_educated_lane_line_pixel_search, perform_blind_lane_line_pixel_search, compute_lane_line_coefficients, compute_curvature_of_lane_lines, compute_vehicle_offset

#globals
calibration_components = None
perspective_transform_components = None #perspective_transform_components[0] is warp_perspective_matrix, perspective_transform_components[1] is unwarp_perspective_matrix
prev_left_lane_line_coeff_queue = None
prev_right_lane_line_coeff_queue = None

#run the pipeline on the provided video
def execute_production_pipeline(my_calibration_components, my_perspective_transform_components):
    #establish ability to set globals
    global calibration_components
    global perspective_transform_components
    global prev_left_lane_line_coeff_queue
    global prev_right_lane_line_coeff_queue

    #set globals
    calibration_components = my_calibration_components
    perspective_transform_components= my_perspective_transform_components
    #initialize queues (storing a max of 10 sets of polynomial coefficients for both the left and right lanes
    prev_left_lane_line_coeff_queue = deque(maxlen=10)
    prev_right_lane_line_coeff_queue = deque(maxlen=10)

    #generate video
    clip_handle = VideoFileClip("test_video/project_video.mp4")
    image_handle = clip_handle.fl_image(process_frame)
    image_handle.write_videofile("output_video/processed_project_video.mp4", audio=False)

#process a frame of video through the pipeline
def process_frame(image):
    
    ###################################
    ## PERFORM DISTORTION CORRECTION ##
    ###################################
    
    #undistort image
    undistorted_image = perform_undistort(image, calibration_components)
    
    ###################################
    ## PERFORM PERSPECTIVE TRANSFORM ##
    ###################################

    #transform perspective (warp) - this will squish the depth of field in the source mapping into the height of the image, 
    #which will make the upper 3/4ths blurry, need to adjust dest_upper* y-values to negative to stretch it out and clear the transformed image up
    #we won't do that as we'll lose right dashes in the 720 pix height of the image frame 
    warped_undistorted_image = perform_perspective_transform(undistorted_image, perspective_transform_components[0])
    
    #######################################
    ## PERFORM COLOR/GRADIENT THRESHOLD  ##
    #######################################

    #apply thresholding to warped image and produce a binary result
    thresholded_warped_undistorted_image = perform_thresholding(warped_undistorted_image)
    
    #############################
    ## PERFORM LANE DETECTION  ##
    #############################
    
    #if this is the very first frame, we must do a blind search for the lane lines
    if ((len(prev_left_lane_line_coeff_queue) == 0) and (len(prev_right_lane_line_coeff_queue) == 0)):
        #map out the left and right lane line pixel locations via windowed search
        left_lane_pixel_coordinates, right_lane_pixel_coordinates, _ = perform_blind_lane_line_pixel_search(thresholded_warped_undistorted_image, return_debug_image=False)    
    else:
        #if we have previous coefficients in the queues, use the latest (rightmost) set as a starting place to accelerate our lane search for this frame
        #map out the left and right lane line pixel coordinates via windowed search using previous polynomials as starting place
        left_lane_pixel_coordinates, right_lane_pixel_coordinates, _ = perform_educated_lane_line_pixel_search(thresholded_warped_undistorted_image, prev_left_lane_line_coeff_queue[-1], prev_right_lane_line_coeff_queue[-1], None, None, return_debug_image=False)
    
    #compute the polynomial coefficients for each lane line using the x and y pixel locations from the mapping function
    #we're fitting (computing coefficients of) a second order polynomial: f(y) = A(y^2) + By + C
    #we're fitting for f(y) rather than f(x), as the lane lines in the warped image are near vertical and may have the same x value for more than one y value 
    left_lane_line_coeff, right_lane_line_coeff = compute_lane_line_coefficients(left_lane_pixel_coordinates, right_lane_pixel_coordinates)
    
    #if we have at least one set of previous left and right lane coefficients stored in the queues
    if ((len(prev_left_lane_line_coeff_queue) > 0) and (len(prev_right_lane_line_coeff_queue) > 0)):
        #compute the percentage difference between the current left and right coefficients sets and latest (rightmost) set of coefficients in each queue 
        left_percent_difference = np.abs(left_lane_line_coeff - prev_left_lane_line_coeff_queue[-1]) / np.mean([left_lane_line_coeff, prev_left_lane_line_coeff_queue[-1]])
        right_percent_difference = np.abs(right_lane_line_coeff - prev_right_lane_line_coeff_queue[-1]) / np.mean([left_lane_line_coeff, prev_right_lane_line_coeff_queue[-1]])
        #if the percent difference between any of the coefficients exceeds 3%, use the latest (rightmost) set of previous coefficients in the queue for each lane line (last added to the queue)
        if (np.any(left_percent_difference > 3) or np.any(right_percent_difference > 3)):
            #pop off the newest (rightmost) set of coefficients from the queue
            left_lane_line_coeff = prev_left_lane_line_coeff_queue.pop()
            right_lane_line_coeff = prev_right_lane_line_coeff_queue.pop()
            
    #append the current coefficients to the queue for use on the next frame
    #the queue will automatically pop off the oldest set of coefficients from each queue if maxlength is reach (that way we only keep 10 sets at all time)
    prev_left_lane_line_coeff_queue.append(left_lane_line_coeff)
    prev_right_lane_line_coeff_queue.append(right_lane_line_coeff)
    
    #smooth the lines (trading off line accuracy for reduced jitter) by taking the mean of the sets of coefficients currently in the queue
    left_lane_line_coeff = np.mean(prev_left_lane_line_coeff_queue, axis=0)
    right_lane_line_coeff = np.mean(prev_right_lane_line_coeff_queue, axis=0)
    
    #generate range of evenly spaced numbers over y interval (0 - 719) matching image height
    y_linespace = np.linspace(0, (thresholded_warped_undistorted_image.shape[0] - 1), thresholded_warped_undistorted_image.shape[0])
    
    #left lane fitted polynomial (f(y) = A(y^2) + By + C)
    left_lane_line_fitted_poly = (left_lane_line_coeff[0] * (y_linespace ** 2)) + (left_lane_line_coeff[1] * y_linespace) + left_lane_line_coeff[2]
    #right lane fitted polynomial (f(y) = A(y^2) + By + C)
    right_lane_line_fitted_poly = (right_lane_line_coeff[0] * (y_linespace ** 2)) + (right_lane_line_coeff[1] * y_linespace) + right_lane_line_coeff[2]
    
    ## compute lane curvature ##
    left_curvature, right_curvature = compute_curvature_of_lane_lines(thresholded_warped_undistorted_image.shape, left_lane_line_fitted_poly, right_lane_line_fitted_poly)
    
    ## compute vehicle offset from center ##
    vehicle_offset = compute_vehicle_offset(thresholded_warped_undistorted_image.shape, left_lane_line_coeff, right_lane_line_coeff)
    
    ########################################
    ## PERFORM PROJECTION BACK ONTO ROAD  ##
    ########################################
    
    #create an image to draw the lines on
    warped_lane = np.zeros_like(warped_undistorted_image).astype(np.uint8)

    #recast the x and y points into usable format for fillPoly and polylines
    pts_left = np.array([np.transpose(np.vstack([left_lane_line_fitted_poly, y_linespace]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane_line_fitted_poly, y_linespace])))])
    pts = np.hstack((pts_left, pts_right))

    #draw the lane onto the warped blank image
    cv2.fillPoly(warped_lane, np.int_([pts]), (152, 251, 152))
    
    #draw fitted lines on image
    cv2.polylines(warped_lane, np.int_([pts_left]), False, color=(189,183,107), thickness=20, lineType=cv2.LINE_AA)
    cv2.polylines(warped_lane, np.int_([pts_right]), False, color=(189,183,107), thickness=20, lineType=cv2.LINE_AA)

    #transform perspective back to original (unwarp)
    warped_to_original_perspective = perform_perspective_transform(warped_lane, perspective_transform_components[1])

    #combine (weight) result with the original image
    projected_lane = cv2.addWeighted(undistorted_image, 1, warped_to_original_perspective, 0.3, 0)

    #add tracking text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(projected_lane, 'Lane curvature: {0:.2f} meters'.format(np.mean([left_curvature, right_curvature])), (20, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(projected_lane, 'Vehicle offset: {0:.2f} meters'.format(vehicle_offset), (20, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    #return processed frame for inclusion in processed video    
    return projected_lane 