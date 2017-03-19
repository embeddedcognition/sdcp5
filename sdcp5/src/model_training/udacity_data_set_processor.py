#########################################################
## AUTHOR: James Beasley                               ##
## DATE: March 18, 2017                                ##
## UDACITY SDC: Project 5 (Vehicle Detection/Tracking) ##
#########################################################

#############
## IMPORTS ##
#############
import csv
import cv2
import uuid

#this data contains real world images with bounding boxes designating the labeled objects (Car, Truck, Pedestrian)
#pre-process udacity data set 1 for usage in training set
#pre-processing steps: load file, crop to bounding box dimensions, resize to 64x64x3, save to training set
with open("udacity_data_set_1/labels.csv") as file_handle:
    #view file rows as a dictionary (based on header row)
    dict_reader = csv.DictReader(file_handle)
    #get current row as dictionary
    for cur_dict_line in dict_reader:
        #get label from cur_row
        cur_label = cur_dict_line["Label"] 
        #ignore pedestrian data (we only care about cars and trucks)
        if (cur_label != "Pedestrian"):
            #load the current frame
            cur_frame = cv2.imread("udacity_data_set_1/" + cur_dict_line["Frame"])
            #extract bounding boxes
            row_min = int(cur_dict_line["xmax"]) #the csv header row is incorrect (this value acually contains the ymin), but we work around it
            row_max = int(cur_dict_line["ymax"])
            col_min = int(cur_dict_line["xmin"])
            col_max = int(cur_dict_line["ymin"]) #the csv header row is incorrect (this value acually contains the xmax), but we work around it
            #crop current frame based on supplied bounding boxes
            cropped_cur_frame = cur_frame[row_min:row_max, col_min:col_max]
            #get cropped current frame size
            cropped_cur_frame_size = cropped_cur_frame.shape 
            #the resize function doesn't like dimensions of size 0
            #there is at least 1 bounding box in the csv that meets this criteria so we need to ignore it
            if ((cropped_cur_frame_size[0] > 0) and (cropped_cur_frame_size[1] > 0)):
                #resize cropped cur frame to 64x64x3
                resized_cropped_cur_frame = cv2.resize(cropped_cur_frame, (64, 64))
                #save the processed frame
                #given the fact that multiple labels can exist in the same frame, we need to ensure file name uniqueness
                cv2.imwrite("training_set/vehicles/objects_extracted_from_udacity_data_set_1/" + str(uuid.uuid4()) + "_" + cur_dict_line["Frame"], resized_cropped_cur_frame)