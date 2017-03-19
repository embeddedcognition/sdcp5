####################################################
## AUTHOR: James Beasley                          ##
## DATE: February 18, 2017                        ##
## UDACITY SDC: Project 4 (Advanced Lane Finding) ##
####################################################

#############
## IMPORTS ##
#############
import cv2

#generate a tuple containing the perspective matrices to warp and unwarp
def generate_perspective_transform_components(src_vertices, dest_vertices):
    #get perspective matrix based on source --> destination point mapping (warp)
    warp_perspective_matrix = cv2.getPerspectiveTransform(src_vertices, dest_vertices)
    #get perspective matrix based on destination --> source point mapping (unwarp)
    unwarp_perspective_matrix = cv2.getPerspectiveTransform(dest_vertices, src_vertices)
    #return perspective matrices
    return (warp_perspective_matrix, unwarp_perspective_matrix)
    
#transform the perspective of the supplied undistorted image using the supplied perspective matrix
def perform_perspective_transform(image, perspective_matrix):
    #warp or unwarp based on perspective matrix supplied
    return cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)