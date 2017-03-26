####################################################
## AUTHOR: James Beasley                          ##
## DATE: February 18, 2017                        ##
## UDACITY SDC: Project 4 (Advanced Lane Finding) ##
####################################################

#############
## IMPORTS ##
#############
import numpy as np
from sdcp4.calibration_processor import generate_calibration_components
from sdcp4.perspective_processor import generate_perspective_transform_components
from test_pipeline import execute_test_pipeline
from production_pipeline import execute_production_pipeline

################################
## PERFORM CAMERA CALIBRATION ##
################################

#set image size for the camera we're working with
camera_image_size = (1280, 720) #(cols, rows)

#inside corner count of chessboard calibration images
num_column_points = 9  #total inside corner points across the x-axis
num_row_points = 6     #total inside corner points across the y-axis

#path to calibration images
path_to_calibration_images = "camera_cal/*.jpg"

#generate calibration componenets used to perform undistort
camera_matrix, distortion_coeff = generate_calibration_components(num_column_points, num_row_points, path_to_calibration_images, camera_image_size)

#package calibration components in a tuple for easy transport
calibration_components = (camera_matrix, distortion_coeff)

################################
## PERSPECTIVE TRANSFORM INIT ##
################################

#set source vertices for region mask
src_upper_left =  (517, 478)
src_upper_right = (762, 478)
src_lower_left = (0, 720)
src_lower_right = (1280, 720)

#set destination vertices (for perspective transform)
dest_upper_left = (0, 0)
dest_upper_right = (1280, 0)
dest_lower_left = (0, 720)
dest_lower_right = (1280, 720)

#package source vertices (points)
src_vertices = np.float32(
    [src_upper_left,
     src_lower_left,
     src_lower_right,
     src_upper_right])

#package destination vertices (points)
dest_vertices = np.float32(
    [dest_upper_left,
     dest_lower_left,
     dest_lower_right,
     dest_upper_right])

#generate perspective transform componenets used to warp/unwarp
warp_perspective_matrix, unwarp_perspective_matrix = generate_perspective_transform_components(src_vertices, dest_vertices)

#package perspective transform components in a tuple for easy transport
perspective_transform_components = (warp_perspective_matrix, unwarp_perspective_matrix)

###################
## TEST PIPELINE ##
###################

#test the execution of the pipeline stages (output from each stage is saved to the output_images directory)  
execute_test_pipeline(calibration_components, perspective_transform_components, (src_upper_left, src_lower_left, src_lower_right, src_upper_right))

#########################
## PRODUCTION PIPELINE ##
#########################

#execute the pipeline (producing a video that is saved to the output_video directory)   
execute_production_pipeline(calibration_components, perspective_transform_components)