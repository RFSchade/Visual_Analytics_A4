#============================================================#
#=============> Logistic Regression Classifier <=============#
#============================================================#

#=====> Import modules
# System tools
import os
import sys
import argparse
sys.path.append(os.getcwd())

# Data tools
import numpy as np
from tqdm import tqdm
import pandas as pd
from random import sample
from itertools import chain

# Cifar-10 data
from tensorflow.keras.datasets import cifar10

# Image manipulation tools
import cv2

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

#=====> Define Functions
# > Load data
def load_data_lr(sample = None, size = 210):
    # Print info
    print("[INFO] Loading data...")
    
    # > load y data 
    filepath = os.path.join("in", "processed_data", "y_data.npy")
    # Load array
    y = np.load(filepath)
    
    # > Load file_list to be certain that X data will be in the same order as y 
    # Get the filepath
    filepath = os.path.join("in", "processed_data", "file_list_img.csv")
    # Reading the filepath 
    file_list = pd.read_csv(filepath)
    
    # Choose to sample or not
    if sample: 
        # Convert y to dataframe to sample
        y_df = pd.DataFrame(y, columns =["label"])
        # Sample and get index
        y_sample = (y_df.groupby("label", as_index=False)
                    .apply(lambda x: x.sample(n=int(sample), replace=False).index)
                    .reset_index(drop=True))
        # Convert 2d list if indexes into 1d
        flatten_list = list(chain.from_iterable(y_sample))
        # Use indexes to find the y values in the sample
        y_relevant = np.array([y[i] for i in flatten_list])
        
        # Define list of files to iretate over
        y_filenames = [file_list["files"].tolist()[i] for i in flatten_list]
    else:
        # Define list of files to iretate over
        y_filenames = file_list["files"].tolist()
        # Define relevant y values
        y_relevant = y
    
    # Define empthy list 
    X = []
    # Iterate over images to load as arrays
    for file in tqdm(y_filenames):
        # Get filepath
        filepath = os.path.join("in", "images", file)
        # Load image
        image = cv2.imread(filepath)
        # Convert to greyscale
        gray = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        # Resize image
        compressed_gray = cv2.resize(gray, (int(size), int(size)), interpolation=cv2.INTER_AREA)
        # Append to list
        X.append(compressed_gray.flatten())

    # Making sure that both X and y are numpy arrays
    X = np.array(X)
    y_relevant = np.array(y_relevant)

    # Splitting data 
    X_train, X_test, y_train, y_test = train_test_split(X, y_relevant,
                                                    random_state=42,
                                                    test_size = 0.2)
    
    return X_train, X_test, y_train, y_test

# > Normalize data
def normalize(X_train, X_test):
    # Scaling the features
    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255
    
    return (X_train_scaled, X_test_scaled)

# > Train model
def train_model(X_train, y_train):
    # Print info
    print("[info] Training model...")
    # Initialyzing model
    clf = LogisticRegression(multi_class="multinomial")
    # Training model 
    clf = LogisticRegression(penalty="none",
                             tol=0.1,
                             solver="saga",
                             multi_class="multinomial").fit(X_train, y_train.ravel()) 
                              
    return clf
    
# > Report
def report(clf, X_test, y_test):
    # Print info 
    print("[info] Reporting results...")
    
    # > Load label names
    # Get the filepath
    filepath = os.path.join("in", "processed_data", "label_names.csv")
    # Reading the filepath 
    label_names = pd.read_csv(filepath)
    
    
    # Predict classification of test data
    y_pred = clf.predict(X_test)
    # Get metrics
    report = metrics.classification_report(y_test, 
                                           y_pred,
                                           target_names=label_names["labels"])
    # Print metrics
    print(report)
    # Save metrics
    outpath = os.path.join("output", "lr_report.txt")
    with open(outpath, "w") as f:
        f.write(report)

# > Custom argument type for useability
def subset_type(x):
    x = int(x)
    if x > 276:
        raise argparse.ArgumentTypeError("Subset argument must be equal to or smaller than 276")
    return x
        
# > Parse arguments
def parse_args(): 
    # Initialize argparse
    ap = argparse.ArgumentParser()
    # Commandline parameters 
    ap.add_argument("-s", "--size", 
                    required=False, 
                    help="Value to convert the image height and width to", 
                    default= 210)
    ap.add_argument("-sub", "--subset",
                    type=subset_type,
                    required=False, 
                    help="Number of values in each category to include in a subset to the data - max 276")
    # Parse argument
    args = vars(ap.parse_args())
    # return list of argumnets 
    return args
        
#=====> Define main()
def main():
    # Get argument
    args = parse_args()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data_lr(args["subset"], args["size"])
    # normalize data
    X_train, X_test = normalize(X_train, X_test)
    # Training model 
    clf = train_model(X_train, y_train)
    # Reporting data 
    report(clf, X_test, y_test)
    
    # Print info 
    print("[INFO] Job complete")

# Run main() function from terminal only
if __name__ == "__main__":
    main()
