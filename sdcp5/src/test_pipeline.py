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
import matplotlib.image as mpimg
from calibration_processor import perform_undistort
from perspective_processor import perform_perspective_transform
from threshold_processor import perform_thresholding
from lane_processor import perform_educated_lane_line_pixel_search, perform_blind_lane_line_pixel_search, compute_lane_line_coefficients, compute_curvature_of_lane_lines, compute_vehicle_offset

#test the pipeline components and produce outputs in the output_images directory
#perspective_transform_components[0] is warp_perspective_matrix, perspective_transform_components[1] is unwarp_perspective_matrix
def execute_test_pipeline(calibration_components, perspective_transform_components, src_vertices):
    
    #############################
    ## TEST CAMERA CALIBRATION ##
    #############################
    
    #test camera calibration by undistorting a test chessboard image
    #load image
    test_chessboard_image = mpimg.imread("camera_cal/calibration1.jpg")
    #undistort image
    undistorted_test_chessboard_image = perform_undistort(test_chessboard_image, calibration_components)
    #save image
    mpimg.imsave("output_images/stage0_undistorted_calibration1.jpg", undistorted_test_chessboard_image)

    #test camera calibration by undistorting a test road image
    #load image
    test_road_image = mpimg.imread("test_images/straight_lines1.jpg")
    #undistort image - this undistorted image will be used to demonstrate the production_pipeline along the way (all outputs will be placed in 'output_images' folder)
    undistorted_test_road_image = perform_undistort(test_road_image, calibration_components)
    #save image
    mpimg.imsave("output_images/stage0_undistorted_straight_lines1.jpg", undistorted_test_road_image)

    ################################
    ## TEST PERSPECTIVE TRANSFORM ##
    ################################

    #drawing parameters
    line_color = [255, 0, 0] #red
    line_thickness = 3

    #draw lines on test road image to display source vertices 
    src_vertices_image = undistorted_test_road_image.copy() #copy as not to affect original image
    cv2.line(src_vertices_image, src_vertices[0], src_vertices[3], line_color, line_thickness)
    cv2.line(src_vertices_image, src_vertices[0], src_vertices[1], line_color, line_thickness)
    cv2.line(src_vertices_image, src_vertices[3], src_vertices[2], line_color, line_thickness)
    cv2.line(src_vertices_image, src_vertices[1], src_vertices[2], line_color, line_thickness)
    #save image
    mpimg.imsave("output_images/stage1_src_vertices_straight_lines1.jpg", src_vertices_image)

    #transform perspective (warp) - this will squish the depth of field in the source mapping into the height of the image, 
    #which will make the upper 3/4ths blurry, need to adjust dest_upper* y-values to negative to stretch it out and clear the transformed image up
    #we won't do that as we'll lose right dashes in the 720 pix height of the image frame 
    warped_undistorted_test_road_image = perform_perspective_transform(undistorted_test_road_image, perspective_transform_components[0])
    #save image
    mpimg.imsave("output_images/stage1_warped_straight_lines1.jpg", warped_undistorted_test_road_image)

    #draw lines on warped test road image to check alignment of lanes
    lane_alignment_warped_undistorted_test_road_image = warped_undistorted_test_road_image.copy() #copy as not to affect original image
    #set lane alignment verticies to check correctness of perspective transform
    lane_alignment_upper_left = (205, 0)
    lane_alignment_upper_right = (1105, 0)
    lane_alignment_lower_left = (205, 720)
    lane_alignment_lower_right = (1105, 720)
    #draw lane alignment lines on warped image
    cv2.line(lane_alignment_warped_undistorted_test_road_image, lane_alignment_upper_left, lane_alignment_upper_right, line_color, line_thickness)
    cv2.line(lane_alignment_warped_undistorted_test_road_image, lane_alignment_upper_left, lane_alignment_lower_left, line_color, line_thickness)
    cv2.line(lane_alignment_warped_undistorted_test_road_image, lane_alignment_upper_right, lane_alignment_lower_right, line_color, line_thickness)
    cv2.line(lane_alignment_warped_undistorted_test_road_image, lane_alignment_lower_left, lane_alignment_lower_right, line_color, line_thickness)
    #save image
    mpimg.imsave("output_images/stage1_lane_alignment_warped_straight_lines1.jpg", lane_alignment_warped_undistorted_test_road_image)

    ####################################
    ## TEST COLOR/GRADIENT THRESHOLD  ##
    ####################################

    #apply thresholding to warped image and produce a binary result
    thresholded_warped_undistorted_test_road_image = perform_thresholding(warped_undistorted_test_road_image)
    
    #export as black and white image (instead of current single channel binary image which would be visualized as blue (representing zeros) and red (representing positive values 1 or 255) 
    #scale to 8-bit (0 - 255) then convert to type = np.uint8
    thresholded_warped_undistorted_test_road_image_scaled = np.uint8((255 * thresholded_warped_undistorted_test_road_image) / np.max(thresholded_warped_undistorted_test_road_image))
    #stack to create final black and white image
    thresholded_warped_undistorted_test_road_image_bw = np.dstack((thresholded_warped_undistorted_test_road_image_scaled, thresholded_warped_undistorted_test_road_image_scaled, thresholded_warped_undistorted_test_road_image_scaled))
    #save image
    mpimg.imsave("output_images/stage2_thresholded_warped_straight_lines1.jpg", thresholded_warped_undistorted_test_road_image_bw)
    
    #########################
    ## TEST LANE DETECTION ##
    #########################
    
    ## BLIND SEARCH ##
    
    #map out the left and right lane line pixel coordinates via windowed search
    left_lane_pixel_coordinates, right_lane_pixel_coordinates, blind_debug_image = perform_blind_lane_line_pixel_search(thresholded_warped_undistorted_test_road_image, return_debug_image=True)
    
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
    #draw lines (on the blind search debug image)
    cv2.polylines(blind_debug_image, np.int_([pts_left]), False, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(blind_debug_image, np.int_([pts_right]), False, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    #save image
    mpimg.imsave("output_images/stage3_blind_search_fitted_polynomials_straight_lines1.jpg", blind_debug_image)
    
    ## EDUCATED SEARCH ##
    
    #map out the left and right lane line pixel coordinates via windowed search
    left_lane_pixel_coordinates, right_lane_pixel_coordinates, educated_debug_image = perform_educated_lane_line_pixel_search(thresholded_warped_undistorted_test_road_image, left_lane_line_coeff, right_lane_line_coeff, left_lane_line_fitted_poly, right_lane_line_fitted_poly, return_debug_image=True)
    
    #compute the polynomial coefficients for each lane line using the x and y pixel locations from the mapping function
    #we're fitting (computing coefficients of) a second order polynomial: f(y) = A(y^2) + By + C
    #we're fitting for f(y) rather than f(x), as the lane lines in the warped image are near vertical and may have the same x value for more than one y value 
    left_lane_line_coeff, right_lane_line_coeff = compute_lane_line_coefficients(left_lane_pixel_coordinates, right_lane_pixel_coordinates)
    
    #left lane fitted polynomial (f(y) = A(y^2) + By + C)
    left_lane_line_fitted_poly = (left_lane_line_coeff[0] * (y_linespace ** 2)) + (left_lane_line_coeff[1] * y_linespace) + left_lane_line_coeff[2]
    #right lane fitted polynomial (f(y) = A(y^2) + By + C)
    right_lane_line_fitted_poly = (right_lane_line_coeff[0] * (y_linespace ** 2)) + (right_lane_line_coeff[1] * y_linespace) + right_lane_line_coeff[2]

    #draw the fitted polynomials on the debug_image and export
    #recast the x and y points into usable format for polylines and fillPoly
    pts_left = np.array([np.transpose(np.vstack([left_lane_line_fitted_poly, y_linespace]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane_line_fitted_poly, y_linespace])))])
    #draw lines (on the educated search debug image)
    cv2.polylines(educated_debug_image, np.int_([pts_left]), False, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(educated_debug_image, np.int_([pts_right]), False, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    #save image
    mpimg.imsave("output_images/stage3_educated_search_fitted_polynomials_straight_lines1.jpg", educated_debug_image)
    
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

    #save image
    mpimg.imsave("output_images/stage4_warped_lane_straight_lines1.jpg", warped_lane)

    #transform perspective back to original (unwarp)
    warped_to_original_perspective = perform_perspective_transform(warped_lane, perspective_transform_components[1])

    #combine (weight) the result with the original image
    projected_lane = cv2.addWeighted(undistorted_test_road_image, 1, warped_to_original_perspective, 0.3, 0)
    
    #add tracking text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(projected_lane, 'Lane curvature: {0:.2f} meters'.format(np.mean([left_curvature, right_curvature])), (20, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(projected_lane, 'Vehicle offset: {0:.2f} meters'.format(vehicle_offset), (20, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    #save image
    mpimg.imsave("output_images/stage4_projected_lane_straight_lines1.jpg", projected_lane)