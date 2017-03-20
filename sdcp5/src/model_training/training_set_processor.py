#########################################################
## AUTHOR: James Beasley                               ##
## DATE: March 18, 2017                                ##
## UDACITY SDC: Project 5 (Vehicle Detection/Tracking) ##
#########################################################

#############
## IMPORTS ##
#############
import cv2
import glob
import numpy as np
import random
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from collections import Counter

#loads a set of training images from a particular path, assigning the supplied label to each image 
def load_training_set(path_to_training_images, label_to_assign):
    X_train = []    #training examples (images)
    y_train = []    #labels: this is binary classification so 0 or 1
    #enumerate image file path list, loading each image, converting to RGB, and adding it to the X_train list 
    for cur_image_file_path in glob.iglob(path_to_training_images, recursive=True):
        #load current image
        bgr_image = cv2.imread(cur_image_file_path)
        #convert current image from BGR (opencv standard) to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        #append loaded image to X_train list
        X_train.append(rgb_image)
        #append supplied label to y_train list
        y_train.append(label_to_assign)
    #return tuple of training images and labels
    return (np.array(X_train), np.array(y_train))

######################################
## GENERATE SYNTHETIC TRAINING DATA ##
######################################

#translate (change position of) training example
#translation matrix found here: http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
def generate_translation_matrix(image):
    #randomly translate x
    translated_x = np.random.uniform(low=-15, high=15)
    #randomly translate y
    translated_y = np.random.uniform(low=-15, high=15)
    #return translation matrix based on above values
    return np.float32([[1, 0, translated_x],[0, 1, translated_y]])

#perform brightness adjustment (brighten or darken)
def perform_brightness_adjustment(image):
    #convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #randomly adjust V channel
    hsv[:, :, 2] = hsv[:, :, 2] * np.random.uniform(low=0.2, high=1.0)
    #convert back to RGB and return
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#randomly translate object (within specified matrix bounds)
def perform_translation(image):
    #get tanslation matrix
    object_transform_matrix = generate_translation_matrix(image)
    #return randomly translated image
    return cv2.warpAffine(image, object_transform_matrix, (image.shape[1], image.shape[0]))

#generate a synthetic example from the supplied training example
def generate_synthetic_training_example(image):
    #list of transformation functions available
    transformation_functions = [perform_translation, perform_brightness_adjustment]
    #choose the number of transformations to perform at random (between 1 and 2)
    num_transformations_to_perform = random.randint(1, len(transformation_functions))
    #perform the number of transformations chosen
    for _ in range(0, num_transformations_to_perform):
        #select a transformation function at random
        selected_transformation_function = random.choice(transformation_functions)           
        #execute the transformation function and return the result
        image = selected_transformation_function(image)
        #ensure each transformation can only be performed once by removing it from the list
        transformation_functions.remove(selected_transformation_function)
    #return transformed image
    return image

#generate synthetic examples for a particular class
def generate_cur_class_synthetic_training_examples(cur_class, X_train, y_train, num_synthetic_examples_to_create):
    #determine the indexes where cur_class exists within the training set
    cur_class_instance_indexes = [index for index, value in enumerate(y_train) if value == cur_class]
    #determine list length for later use
    num_cur_class_instance_indexes = len(cur_class_instance_indexes)
    #lists to house the synthetic training examples for the class we're working with
    X_train_synthetic_cur_class = [] 
    y_train_synthetic_cur_class = [] 
    #generate synthetic examples to augment and balance cur_class
    for _ in range(0, num_synthetic_examples_to_create):
        #randomly select an index within the list of instances of cur_class
        random_index = random.randint(0, (num_cur_class_instance_indexes - 1))
        #select the randomly chosen index from the list of cur_class instance indexes
        random_cur_class_instance_index = cur_class_instance_indexes[random_index]
        #create a synthetic version of the example at that index from the training set
        X_train_synthetic_cur_class.append(generate_synthetic_training_example(X_train[random_cur_class_instance_index]))
        y_train_synthetic_cur_class.append(y_train[random_cur_class_instance_index])
    return (X_train_synthetic_cur_class, y_train_synthetic_cur_class)
        
#generate synthetic data to balance and augment the training set
def generate_synthetic_training_examples(X_train, y_train, num_class_instances_target):
    #get the distinct set of classes in y_train
    distinct_y_train_classes = set(y_train)
    #lists to hold synthetic training data & labels
    X_train_synthetic = []
    y_train_synthetic = []
    #enumerate the distinct set of classes in the training set and augment/balance data for each
    for cur_class in distinct_y_train_classes:
        #get the instance count of cur_class from y_train
        num_cur_class_instances = list(y_train).count(cur_class)
        #determine the number of synthetic examples to create to both augment and balance cur_class
        num_synthetic_examples_to_create = num_class_instances_target - num_cur_class_instances
        #generate synthetic examples for cur_class
        X_train_synthetic_cur_class, y_train_synthetic_cur_class = generate_cur_class_synthetic_training_examples(cur_class, X_train, y_train, num_synthetic_examples_to_create)
        #append synthetic cur_class examples to total synthetic example set
        X_train_synthetic.append(X_train_synthetic_cur_class)
        y_train_synthetic.append(y_train_synthetic_cur_class)
    #concatenate all of the 4d arrays within the X_train_synthetic and 1d arrays within y_train_synthetic lists 
    return (np.concatenate(X_train_synthetic), np.concatenate(y_train_synthetic))

#load the project data sets
#vehicles (the '**' pattern designates recursion)
X_train_vehicles, y_train_vehicles = load_training_set("model_training/training_set/vehicles/project_vehicles/**/*.png", label_to_assign=1)
#non-vehicles (the '**' pattern designates recursion)
X_train_non_vehicles, y_train_non_vehicles = load_training_set("model_training/training_set/non_vehicles/project_non_vehicles/**/*.png", label_to_assign=0)

print("Num project vehicle images:", X_train_vehicles.shape)
print("Num project vehicle labels:", y_train_vehicles.shape)
print("Num project non-vehicle images:", X_train_non_vehicles.shape)
print("Num project non-vehicle labels:", y_train_non_vehicles.shape)

#stack the two data sets into a single training set
X_train = np.vstack((X_train_vehicles, X_train_non_vehicles))
y_train = np.hstack((y_train_vehicles, y_train_non_vehicles))

#shuffle the training set before carving off the test set
X_train, y_train = shuffle(X_train, y_train)

#carve out a portion of the training set to use for model validation
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

print()
print("Num X_train images:", X_train.shape)
print("Num y_train labels:", y_train.shape)
print("Num X_test images:", X_test.shape)
print("Num y_test labels:", y_test.shape)

#load the udacity data set
#vehicles (the '**' pattern designates recursion)
X_train_udacity_vehicles, y_train_udacity_vehicles = load_training_set("model_training/training_set/vehicles/objects_extracted_from_udacity_data_set_1/**/*.jpg", label_to_assign=1)

print("Num udacity vehicle images:", X_train_udacity_vehicles.shape)
print("Num udacity vehicle labels:", y_train_udacity_vehicles.shape)

#now that we have the udacity data set loaded, we should add it to the existing training set

#stack the two data sets into a single training set
X_train = np.vstack((X_train, X_train_udacity_vehicles))
y_train = np.hstack((y_train, y_train_udacity_vehicles))

#shuffle the training set again
X_train, y_train = shuffle(X_train, y_train)

print()
print("Num X_train images:", X_train.shape)
print("Num y_train labels:", y_train.shape)
        
#the minimum number of additional examples per class we'd like to add, 
#due to class imbalance more will be added as well to balance all classes
num_augmentation_instances_target = 100
#get a dictionary with the tally for each class in the data set (in descending order)
num_class_instances_by_class = Counter(y_train)
#extract the majority class count
num_majority_class_instances = (num_class_instances_by_class.most_common(1))[0][1]
#set the target class count for each class in the data set
num_class_instances_target = num_majority_class_instances + num_augmentation_instances_target
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

#get a dictionary with the tally for each class in the data set (in descending order)
num_class_instances_by_class = Counter(y_train)
print("Number of class instances by class:")
print(num_class_instances_by_class)
print()
#extract majority class
majority_class = (num_class_instances_by_class.most_common(1))[0][0]
print("Majority class:", majority_class)
#extract the majority class count
num_majority_class_instances = (num_class_instances_by_class.most_common(1))[0][1]
print("Majority class count:", num_majority_class_instances)

#number of training examples
n_train = len(X_train)
#number of testing examples
n_test = len(X_test)
#image shape
image_shape = (X_train[0]).shape
#unique classes/labels in the data set
n_classes = len(set(y_train))

print("Number of training examples:", n_train)
print("Number of testing examples:", n_test)
print("Image data shape:", image_shape)
print("Number of classes:", n_classes)

#generate synthetic training examples for each class based on existing examples within each class
X_train_synthetic, y_train_synthetic = generate_synthetic_training_examples(X_train, y_train, num_class_instances_target)
print("X_train_synthetic shape: ", X_train_synthetic.shape)
print("y_train_synthetic shape: ", y_train_synthetic.shape)

#add additional training examples to existing training data to balance and augment
#converting each to float here as if we wait until after they're combined we'll have memory errors due to the size of the set
X_train = np.append(X_train, X_train_synthetic, axis=0)
y_train = np.append(y_train, y_train_synthetic, axis=0)
print("X_train + X_synthetic shape: ", X_train.shape)
print("y_train + y_synthetic shape: ", y_train.shape)

num_class_instances_by_class = Counter(y_train)
print(num_class_instances_by_class)

#shuffle the training set again
X_train, y_train = shuffle(X_train, y_train)

#pickel the data set

#pickled file names
training_set_file = "model_training/pickled_objects/training_set.p"
test_set_file = "model_training/pickled_objects/test_set.p"

#embed data in dictionary
training_set_dict = {"features": X_train, "labels": y_train}
test_set_dict = {"features": X_test, "labels": y_test}

#pickle training data
with open(training_set_file, mode="wb") as f:
    pickle.dump(training_set_dict, f)
#pickle test data
with open(test_set_file, mode="wb") as f:
    pickle.dump(test_set_dict, f)