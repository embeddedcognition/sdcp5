#########################################################
## AUTHOR: James Beasley                               ##
## DATE: March 18, 2017                                ##
## UDACITY SDC: Project 5 (Vehicle Detection/Tracking) ##
#########################################################

#############
## IMPORTS ##
#############
import cv2
import time
import pickle
import numpy as np
from collections import Counter
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from model_training.feature_processor import perform_hog_feature_extraction, perform_spatial_reduction, compute_pixel_intensity_frequency_distribution

###############
## FUNCTIONS ##
###############

#extract color, shape, and gradient features from raw pixel intensities for each image in the supplied data set, returning a list of feature vectors (one for each image)
def perform_feature_extraction(X, spatial_reduction_size, pixel_intensity_fd_bins, hog_orientation_bins, hog_pixels_per_cell, hog_cells_per_block):
    #local vars
    X_features = [] #each row contains a feature vector for the associated image in the supplied training set
    #enumerate the rgb images in the data set and extract features for each and add to the X_train_features list
    for cur_rgb_image in X:
        #convert from rgb to ycrcb color space (achieved better results using this color space) 
        cur_ycrcb_image = cv2.cvtColor(cur_rgb_image, cv2.COLOR_RGB2YCR_CB)
        ## EXTRACT RAW PIXEL INTENSITY FEATURES (TARGETING COLOR & SHAPE) ##
        #reduce resolution while still preserving relevant features
        #raw pixel intensities (even when down-sampled) reveal color and shape characteristics
        raw_pixel_intensity_features = perform_spatial_reduction(cur_ycrcb_image, spatial_reduction_size)
        #compute pixel intensity frequency distribution (all three channels will be computed separately and concatenated
        #computing a frequency distribution of the raw pixel intensities reveals only color characteristics
        raw_pixel_intensity_fd_features = compute_pixel_intensity_frequency_distribution(cur_ycrcb_image, pixel_intensity_fd_bins)
        ## EXTRACT GRADIENT OF RAW PIXEL INTENSITY FEATURES (TARGETING SHAPE) ##
        hog_features_by_channel = []
        #process each channel of the the cur ycrcb image separately
        for channel in range(cur_ycrcb_image.shape[2]):
            #perform histogram of oriented gradients (hog) operation, returning a feature vector
            hog_features_by_channel.append(perform_hog_feature_extraction(cur_ycrcb_image[:, :, channel], hog_orientation_bins, hog_pixels_per_cell, hog_cells_per_block, True, False))
        #stack the rows (one for each channel) of the hog feature list horizontally into one long feature vector
        raw_pixel_intensity_gradient_features = np.ravel(hog_features_by_channel)  
        #concatenate the 3 feature vectors and add them 
        X_features.append(np.concatenate((raw_pixel_intensity_features, raw_pixel_intensity_fd_features, raw_pixel_intensity_gradient_features)))
    #return features
    return X_features

######################################################
## DE-PICKLE, EXTRACT, SHUFFLE TRAINING & TEST SETS ##
######################################################

print()
print("De-pickling training and test data sets...")
print()

#pickled file names
training_set_file = "pickled_objects/training_set.p"
test_set_file = "pickled_objects/test_set.p"

#import pickled data files
with open(training_set_file, mode="rb") as f:
    training_set = pickle.load(f)
with open(test_set_file, mode="rb") as f:
    test_set = pickle.load(f)

#unpack training and test sets    
X_train, y_train = training_set["features"], training_set["labels"]
X_test, y_test = test_set["features"], test_set["labels"]

#shuffle the training set
X_train, y_train = shuffle(X_train, y_train)

print("Num X_train images:", X_train.shape)
print("Num y_train labels:", y_train.shape)
print("Num X_test images:", X_test.shape)
print("Num y_test labels:", y_test.shape)

#get a dictionary with the tally for each class in the data sets (in descending order)
num_class_instances_by_class_train = Counter(y_train)
num_class_instances_by_class_test = Counter(y_test)
print()
print("Num class instances by class:")
print("X_train:", num_class_instances_by_class_train)
print("X_test:", num_class_instances_by_class_test)

##################################################################
## PERFORM FEATURE EXTRACTION & SCALING ON TRAINING & TEST SETS ##
##################################################################

print()
print("Extracting features from training and test data sets...")
print()

#hyperparameters
spatial_reduction_size = 32   #reduce the training images from 64x64 to 32x32 resolution (smaller feature vector but still retains useful shape and color information)
pixel_intensity_fd_bins = 64  #number of bins to use to compute raw pixel intensity frequency distribution
hog_orientation_bins =  9     #number of orientation bins to use in hog feature extraction
hog_pixels_per_cell = 8       #number of pixels per cell to use in hog feature extraction
hog_cells_per_block = 2       #number of cells per block to use in hog feature extraction

#extract features from training and test sets
X_train_features = perform_feature_extraction(X_train, spatial_reduction_size, pixel_intensity_fd_bins, hog_orientation_bins, hog_pixels_per_cell, hog_cells_per_block)
X_test_features = perform_feature_extraction(X_test, spatial_reduction_size, pixel_intensity_fd_bins, hog_orientation_bins, hog_pixels_per_cell, hog_cells_per_block)

#convert to array and cast to float64 (expected by scaler function)
X_train_features = np.array(X_train_features).astype(np.float64)
X_test_features = np.array(X_test_features).astype(np.float64)

#fit a per-column scaler to the training feature set
X_train_feature_scaler = StandardScaler().fit(X_train_features)

#scale the training and test feature sets based on the scaler fitted from the training set
X_train_features_scaled = X_train_feature_scaler.transform(X_train_features)
X_test_features_scaled = X_train_feature_scaler.transform(X_test_features)

########################
## TRAIN & TEST MODEL ##
########################

print("Training model...")
print()

#create instance of linear support vector classifier 
support_vector_classifier = LinearSVC()

#capture start time
start_time=time.time()

#start training
support_vector_classifier.fit(X_train_features_scaled, y_train)

#capture end time
end_time = time.time()

print("Seconds to train model:", round(end_time - start_time, 2))
print()

#display accuracy of the trained model on the scaled test feature set
print("Accuracy of trained model on scaled test feature set:", round(support_vector_classifier.score(X_test_features_scaled, y_test), 4))
print()

## check the prediction time for a single sample ##
sample_size = 10

#capture start time
start_time=time.time()

print("Trained model predicts:" , support_vector_classifier.predict(X_test_features_scaled[0:sample_size]))
print("For a sample size of", sample_size, "labels:", y_test[0:sample_size])
print()

#capture end time
end_time = time.time()

print("Seconds for trained model to predict that sample size:", round(end_time - start_time, 2))

##########################
## PICKLE TRAINED MODEL ##
##########################

print()
print("Pickling fitted model and scaler...")
print()

#pickled file name
trained_model_file = "pickled_objects/trained_model.p"

#embed objects in dictionary
trained_model_dict = {"model": support_vector_classifier, "scaler": X_train_feature_scaler}

#pickle the model and the fitted scaler
with open(trained_model_file, mode="wb") as f:
    pickle.dump(trained_model_dict, f)

####################################################
## DE-PICKLE, EXTRACT, TEST FITTED MODEL & SCALER ##
####################################################

print("Reloading model and scaler to test integrity...")
print()
    
#load pickled objects
with open(trained_model_file, mode="rb") as f:
    loaded_trained_model_dict = pickle.load(f)

#extract fitted model and scalar for testing    
loaded_model, loaded_scaler = loaded_trained_model_dict["model"], loaded_trained_model_dict["scaler"] 

#scale the X_test_features with the loaded scaler to verify integrity 
X_test_features_scaled = loaded_scaler.transform(X_test_features)

#display accuracy of the trained model on the scaled test feature set
print("Accuracy of loaded model on scaled test feature set:", round(loaded_model.score(X_test_features_scaled, y_test), 4))
print()

#capture start time
start_time=time.time()

print("Loaded model predicts:" , loaded_model.predict(X_test_features_scaled[0:sample_size]))
print("For a sample size of", sample_size, "labels:", y_test[0:sample_size])

#capture end time
end_time = time.time()

print()
print("Done.")
print()