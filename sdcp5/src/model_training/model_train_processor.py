

import cv2
import time
import pickle
import numpy as np
from collections import Counter
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

#pickled file names
training_set_file = "model_training/pickled_objects/training_set.p"
test_set_file = "model_training/pickled_objects/test_set.p"

#import pickled data files
with open(training_set_file, mode="rb") as f:
    training_set = pickle.load(f)
with open(test_set_file, mode="rb") as f:
    test_set = pickle.load(f)
    
X_train, y_train = training_set["features"], training_set["labels"]
print(X_train.shape)
print(y_train.shape)

X_test, y_test = test_set["features"], test_set["labels"]
print(X_test.shape)
print(y_test.shape)

#shuffle the training set
X_train, y_train = shuffle(X_train, y_train)

#get a dictionary with the tally for each class in the data set (in descending order)
num_class_instances_by_class = Counter(y_train)
print("Number of training class instances by class:")
print(num_class_instances_by_class)
print()
#extract majority class
majority_class = (num_class_instances_by_class.most_common(1))[0][0]
print("Majority class:", majority_class)
#extract the majority class count
num_majority_class_instances = (num_class_instances_by_class.most_common(1))[0][1]
print("Majority class count:", num_majority_class_instances)

#get a dictionary with the tally for each class in the data set (in descending order)
num_test_class_instances_by_class = Counter(y_test)
print("Number of test class instances by class:")
print(num_test_class_instances_by_class)
print()
#extract majority class
majority_test_class = (num_test_class_instances_by_class.most_common(1))[0][0]
print("Majority class:", majority_test_class)
#extract the majority class count
num_majority_test_class_instances = (num_test_class_instances_by_class.most_common(1))[0][1]
print("Majority class count:", num_majority_test_class_instances)

# Define a function to return HOG features and visualization
def compute_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

#reduce resolution while still preserving relevant features
#down-sample the image
#rename to perform_down_sample
# Define a function to compute binned color features  
def perform_spatial_binning(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

#compute pixel intensity frequency distribution
# Define a function to compute color histogram features  
def compute_pixel_intensity_histograms(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


#def extract_gradient_features(X_train):

    
def extract_features(X_train, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block):
    features = [] #feature vector to return
    #enumerate the rgb images in the training set and extract color features
    for cur_image in X_train:
        #convert to ycrcb color space
        feature_image = cv2.cvtColor(cur_image, cv2.COLOR_RGB2YCR_CB)
        ## EXTRACT COLOR FEATURES ##
        #reduce resolution while still preserving relevant features
        spatial_features = perform_spatial_binning(feature_image, size=spatial_size)
        #compute histograms of pixel intensity
        hist_features = compute_pixel_intensity_histograms(feature_image, nbins=hist_bins)
        #combine features
        color_features = np.concatenate((spatial_features, hist_features))
        ## EXTRACT GRADIENT FEATURES ##
        #compute histogram of oriented gradients (hog)
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(compute_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
        gradient_features = np.ravel(hog_features)  
        # Append the new feature vector to the features list
        features.append(np.concatenate((color_features, gradient_features)))
    #return features
    return features
        
#try 8, 16, or 24 on spatial
#16
spatial = 32
#try 32 or 48 on histbin
#32

#saw a dip inperformance when bins was less than 32 and greater than ...
#128
hist_bins = 64

orient = 9
pix_per_cell = 8
cell_per_block = 2

X_train_features = extract_features(X_train, (spatial, spatial), hist_bins, orient, pix_per_cell, cell_per_block)

#convert to array and cast to float64
X_train_features = np.array(X_train_features).astype(np.float64)
# Fit a per-column scaler
X_train_feature_scaler = StandardScaler().fit(X_train_features)
# Apply the scaler to X
X_train_features_scaled = X_train_feature_scaler.transform(X_train_features)

X_test_features = extract_features(X_test, (spatial, spatial), hist_bins, orient, pix_per_cell, cell_per_block)
X_test_features = np.array(X_test_features).astype(np.float64)                        
#transform using the scaler fitted from the training set
X_test_features_scaled = X_train_feature_scaler.transform(X_test_features)

# Use a linear SVC 
support_vector_classifier = LinearSVC()
# Check the training time for the SVC
t=time.time()
support_vector_classifier.fit(X_train_features_scaled, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(support_vector_classifier.score(X_test_features_scaled, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', support_vector_classifier.predict(X_test_features_scaled[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

#pickled file names
trained_model_file = "model_training/pickled_objects/trained_model_v9.p"

#embed data in dictionary
trained_model_dict = {"model": support_vector_classifier, "scaler": X_train_feature_scaler}

#pickle model
with open(trained_model_file, mode="wb") as f:
    pickle.dump(trained_model_dict, f)
    
#import pickled data files
with open(trained_model_file, mode="rb") as f:
    loaded_trained_model_dict = pickle.load(f)
    
loaded_model, loaded_scaler = loaded_trained_model_dict["model"], loaded_trained_model_dict["scaler"] 

# Check the score of the SVC
X_test_features_scaled = loaded_scaler.transform(X_test_features)
print('Test Accuracy of SVC = ', round(loaded_model.score(X_test_features_scaled, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', loaded_model.predict(X_test_features_scaled[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')