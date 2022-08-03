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
  
    # drop rows that contain the partial string ".GIF"
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

# > Saving image data as arrays to save time when loading
def save_to_arrays(y, average, size, window):
    # Print info 
    print("[INFO] Reshaping data...")
    
    # Define list of filenames
    y_filenames = sorted(y["filename"].tolist())
    # Define empthy list
    file_list_npy = []
    file_list_img = []

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
            # Get outpath for npy file 
            y_name = file.split(".")[0]
            outpath_nn = os.path.join("in", "np_arrays", f"{y_name}.npy")
            # Save to npy file
            np.save(outpath_nn, img_to_array(resized_color))
            # Add new filename to file list 
            file_list_npy.append(f"{y_name}.npy")
            file_list_img.append(file)
    
    # Remove files in the y data that has been filtered away in the X data
    y_filtered = y[y.filename.isin(file_list_img)]
    
    return file_list_npy, file_list_img, y_filtered

# > Convert format of data for saving
def conv_format(y, file_list_npy, file_list_img):
    # Print Info 
    print("[INFO] Changing the data format...")
    # Factorize y 
    y_array = np.array(pd.factorize(y['category'])[0])
    # Reshape y 
    y_reshape = np.reshape(y_array, (len(y_array),1))
    # Get label names 
    label_names = list(pd.factorize(y['category'])[1])
    # Convert to dataframe for saving
    label_names_df = pd.DataFrame(label_names, columns =["labels"])
    
    # Convert file_list to dataframe
    file_list_npy_df = pd.DataFrame(file_list_npy, columns =["files"])
    file_list_img_df = pd.DataFrame(file_list_img, columns =["files"])
    
    return y_reshape, label_names_df, file_list_npy_df, file_list_img_df

# > Save data 
def save(y_reshape, label_names_df, file_list_npy_df, file_list_img_df):
    # Print info 
    print("[INFO] Saving lists of images...")
    # > Save file list (npy)
    outpath_fl_npy = os.path.join("in", "processed_data", "file_list_npy.csv")
    # Save data
    file_list_npy_df.to_csv(outpath_fl_npy, index=False)
    # > Save file list (img)
    outpath_fl_img = os.path.join("in", "processed_data", "file_list_img.csv")
    # Save data
    file_list_img_df.to_csv(outpath_fl_img, index=False)
    
    # > Save label data
    # Print info 
    print("[INFO] Saving label data...")
    # Get outpath
    outpath_y = os.path.join("in", "processed_data", "y_data.npy")
    # Save to npy file
    np.save(outpath_y, y_reshape)
    
    # Save label names
    # Print info 
    print("[INFO] Saving label names...")
    # Get outpath
    outpath_labels = os.path.join("in", "processed_data", "label_names.csv")
    # Save to csv file
    label_names_df.to_csv(outpath_labels, index=False)

# > Parse arguments
def parse_args(): 
    # Initialize argparse
    ap = argparse.ArgumentParser()
    # Commandline parameters 
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
    # Get arguments
    args = parse_args()
    
    # Load data 
    y = load_y()
    # Clean data 
    y_clean = clean_y(y)
    # Reshape data 
    file_list_npy, file_list_img, y_filtered = save_to_arrays(y_clean, int(args["average"]), 210, int(args["window"]))
    # Convert to appropriate formats 
    y_reshape, label_names_df, file_list_npy_df, file_list_img_df = conv_format(y_filtered, file_list_npy, file_list_img)
    # Save data 
    save(y_reshape, label_names_df, file_list_npy_df, file_list_img_df)
    
    # Print info 
    print("[INFO] Job complete")
    
# Run main() function from terminal only
if __name__ == "__main__":
    main()