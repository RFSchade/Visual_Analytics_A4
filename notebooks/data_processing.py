#=============================================#
#=============> Data processing <=============#
#=============================================#

#=====> Import modules
# System tools
import os
import sys 

# Data tools
import pandas as pd
import numpy as np
from numpy import savetxt
import statistics
 
# Function tools
import argparse
from tqdm import tqdm
    
# Image tools 
import cv2

# Tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#=====> Define global variabels

#=====> Define functions
# > Load data 
def load_y():
    # Print info 
    print("[INFO] Loading data...")
    # > Load test labels
    # Get the filepath
    filepath_test = os.path.join("in", "annotation", "test.txt")
    # Load data
    y_test = pd.read_csv(filepath_test, header=None, names = ("filename", "category"))
    
    # > Load training labels
    # Get the filepath
    filepath_train = os.path.join("in", "annotation", "train.txt")
    # Load data
    y_train = pd.read_csv(filepath_train, header=None, names = ("filename", "category"))
    
    # Combine data 
    y = pd.concat([y_train, y_test], axis=0)
    
    return y

# > Clean label data 
def clean_y(y):
    # Print info 
    print("[INFO] Cleaning data")
    # Removing duplicates
    y_rm_dup = y.drop_duplicates(subset=["filename"])  

    # > Remove GIF files from dataframe
    # identify partial string
    discard = [".GIF"]
  
    # drop rows that contain the partial string "Sci"
    y_no_gif = y_rm_dup[~y_rm_dup.filename.str.contains('|'.join(discard))]

    # > Remiving data that for some reason turned out to be problematic
    # defining list
    problematic = ["gMap_256.jpg", 
                   "gMap_549.jpg", 
                   "gMap_559.jpg", 
                   "gMap_724.jpeg", 
                   "gMap_725.jpeg", 
                   "gMap_726.jpeg",  
                   "gMap_727.jpeg",
                   "surfaceP_8.png"]
    # Remove rows
    y_clean = y_no_gif[~y_no_gif.filename.isin(problematic)]
    
    return y_clean

# > Remove outliers
def rem_outliers(y, median, window):
    # The 'median' argument is the median of the image heights 
    # the 'window' argument is the standatd deviation of the image heoghts times 3
    
    # > Load size metadata
    # Get the filepath
    filepath = os.path.join("in", "img_size_df.csv")
    # Reading the filepath 
    sizes = pd.read_csv(filepath)
    
    # Find outliers
    outliers = sizes.loc[abs(sizes["height"] - median) >= window]
    
    # Remove rows 
    outliers_removed = y[~y.filename.isin(outliers["filename"])]
    
    return outliers_removed

# > Reshape data
def reshape_data(y, average, size, window):
    # Print info 
    print("[INFO] Reshaping data...")
    
    # Define list of filenames
    y_filenames = sorted(y["filename"].tolist())
    # Define empthy list
    X_logistic = []
    X_nn = []

    # Iterate through files
    for file in tqdm(y_filenames):
        # Get filepath
        filepath = os.path.join("in", "images", file)
        # Load image
        image = cv2.imread(filepath)
        
        # Filtering outliers
        if abs(image.shape[0] - average) >= window:
            pass
        else:
            # > Create data for neural network 
            # Resize images
            resized_color = cv2.resize(image, (int(size), int(size)), interpolation=cv2.INTER_AREA)
            # Append array to list 
            X_nn.append(img_to_array(resized_color))
    
            # > Create dataset for logistic regression
            # Convert to greyscale
            gray = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            # Resize image
            compressed_gray = cv2.resize(gray, (int(size), int(size)), interpolation=cv2.INTER_AREA)
            # Flatten image and append to list 
            X_logistic.append(compressed_gray.flatten())
    
    # Convert logistic regression data to dataframe
    X_logistic_df = pd.DataFrame(X_logistic)
    # Convert neural network data to array 
    X_nn_array = np.array(X_nn)
    # Factorize y 
    y_array = np.array(pd.factorize(y['category'])[0])
    # Reshape y 
    y_reshape = np.reshape(y_array, (len(y_array),1))
    # Get label names 
    label_names = list(pd.factorize(y['category'])[1])
    # Convert to dataframe for saving
    label_names_df = pd.DataFrame(label_names, columns =["labels"])
    
    return X_logistic_df, X_nn_array, y_reshape, label_names_df

# Save data 
def save(X_logistic, X_nn, y_array, label_names):
    # Print info 
    print("[INFO] Saving data...")
    
    # > Save logistic regression data
    # Print info 
    print("[INFO] Logistic regression data...")
    
    # Get outpath 
    outpath_lr = os.path.join("in", "processed_data", "lr_data.csv")
    # Save data
    X_logistic.to_csv(outpath_lr, index=False)
    
    # > Save neural network data 
    # Print info 
    print("[INFO] Neural network data...")
    
    # Get outpath
    outpath_nn = os.path.join("in", "processed_data", "nn_data.npy")
    # Save to npy file
    np.save(outpath_nn, X_nn)
    
    # > Save label data
    # Print info 
    print("[INFO] Label data...")
    
    outpath_y = os.path.join("in", "processed_data", "y_data.npy")
    # Save to npy file
    np.save(outpath_y, y_array)
    
    # Save label names
    # Print info 
    print("[INFO] Label names...")
    
    outpath_labels = os.path.join("in", "processed_data", "label_names.csv")
    # Save to csv file
    label_names.to_csv(outpath_labels, index=False)

# > Parse arguments
def parse_args(): 
    # Initialize argparse
    ap = argparse.ArgumentParser()
    # Commandline parameters 
    ap.add_argument("-s", "--size", 
                    required=False, 
                    help="Value to convert the image height and width to", 
                    default= 210)
    ap.add_argument("-a", "--average", 
                    required=False, 
                    help="Average or target value to serve as a baseline for image height", 
                    default= 210)
    ap.add_argument("-w", "--window", 
                    required=False, 
                    help="Window to be added and subtracted from the average value - values outside the window will be removed", 
                    default= 745)
    # Parse argument
    args = vars(ap.parse_args())
    # return list of argumnets 
    return args

#=====> Define main()
def main():
    # Get argument
    args = parse_args()
    
    # Load data 
    y = load_y()
    # Clean data 
    y_clean = clean_y(y)
    # Remove outliers
    # rem_y = rem_outliers(y_clean, int(args["average"]), int(args["window"]))
    # Reshape data 
    X_logistic, X_nn, y_array, label_names = reshape_data(y_clean, int(args["average"]), int(args["size"]), int(args["window"]))
    # Save data 
    save(X_logistic, X_nn, y_array, label_names)
    
    # Print info 
    print("[INFO] Job complete")
    
# Run main() function from terminal only
if __name__ == "__main__":
    main()
