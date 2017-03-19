####################################################
## AUTHOR: James Beasley                          ##
## DATE: February 18, 2017                        ##
## UDACITY SDC: Project 4 (Advanced Lane Finding) ##
####################################################

#############
## IMPORTS ##
#############
import numpy as np
import matplotlib.image as mpimg
import glob
import cv2

#generate calibration camera matrix and distortion coefficients based on supplied chessboard dimensions and chessboard calibration images
#http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
def generate_calibration_components(num_column_points, num_row_points, path_to_calibration_images, camera_image_size): 
    #lists to store object/image points for all calibration images
    calibration_object_points = [] #3d points in real world space
    calibration_image_points = []  #2d points on the image plane
    #a matrix of 3d coordinate values (each row holds an (x, y, z) point with each column being x, y, or z)
    #z will stay 0 since the chessboard is a plane, but we'll generate the x and y coordinates automatically (must be float coordinate points)
    calibration_object_points_template = np.mgrid[0:num_column_points, 0:num_row_points, 0:1].T.reshape(-1, 3).astype(np.float32)
    #load calibration image file path list (file paths to images of chessboards to calibrate from) 
    calibration_image_file_path_list = glob.glob(path_to_calibration_images)
    #enumerate calibration image file path list, loading each image, converting to grayscale, then retrieving inside corner points
    for cur_calibration_image_file_path in calibration_image_file_path_list:
        #load image located at cur_calibration_image_file_path
        cur_calibration_image = mpimg.imread(cur_calibration_image_file_path)
        #convert image to grayscale
        cur_calibration_image_grayscale = cv2.cvtColor(cur_calibration_image, cv2.COLOR_RGB2GRAY)
        #find image points (inside corners of chessboard) for cur_calibration_image_grayscale
        allcornersfound, corners = cv2.findChessboardCorners(cur_calibration_image_grayscale, (num_column_points, num_row_points), None)
        #if all internal corners where found (valid chessboard pattern displaying all internal corners)
        if (allcornersfound):
            #add image points found for cur_calibration_image_grayscale to image_points list
            calibration_image_points.append(corners)
            #add associated objects points for this calibrations image (same for all calibration images)
            calibration_object_points.append(calibration_object_points_template)
    #derive camera matrix (needed to transform 3d object points to 2d image points) and distortion coefficients
    #based on image and object points derived from calibration images taken on that same camera
    _, camera_matrix, distortion_coeff, _, _ = cv2.calibrateCamera(calibration_object_points, calibration_image_points, camera_image_size, None, None)
    #return components
    return (camera_matrix, distortion_coeff)

#transform an image to compensate for radial and tangential lens distortion
#calibration_components[0] is camera_matrix, calibration_components[1] is distortion_coeff 
def perform_undistort(image, calibration_components):
    #return the undistorted image
    return cv2.undistort(image, calibration_components[0], calibration_components[1])