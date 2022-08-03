#================================================================#
#=============> VGG16 feature extraction CNN Model <=============#
#================================================================#

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

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report

# for plotting
import matplotlib.pyplot as plt

#=====> Define global variables
# Epochs
EPOCHS = 3

#=====> Define functions
# > Load data
def load_data_nn(sample = None):
    # Print info
    print("[INFO] Loading data...")
    
    # > load y data 
    filepath = os.path.join("in", "processed_data", "y_data.npy")
    # Load array
    y = np.load(filepath)
    
    # > Load file_list to be certain that X data will be in the same order as y 
    # Get the filepath
    filepath = os.path.join("in", "processed_data", "file_list_npy.csv")
    # Reading the filepath 
    file_list = pd.read_csv(filepath)
    
    # Choose to sample or not
    if sample: 
        # Convert y to dataframe to sample
        y_df = pd.DataFrame(y, columns =["label"])
        # Sample and get index
        y_sample = (y_df.groupby("label", as_index=False)
                    .apply(lambda x: x.sample(n=sample, replace=False).index)
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
        # Get filepath for image
        filepath = os.path.join("in", "np_arrays", file)
        # Load array
        loaded_array = np.load(filepath)
        # Append to list
        X.append(loaded_array)

    # Making sure that both X and y are numpy arrays
    X = np.array(X)
    y_relevant = np.array(y_relevant)

    # Splitting data 
    X_train, X_test, y_train, y_test = train_test_split(X, y_relevant,
                                                    random_state=42,
                                                    test_size = 0.2)
    
    return X_train, X_test, y_train, y_test

# > Normalize data
def normalize(X_train, X_test, y_train, y_test):
    # Normalize data 
    X_train = X_train/255
    X_test = X_test/255
    # Create label encodings 
    # Initialize label names
    lb = LabelBinarizer ()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    
    return X_train, X_test, y_train, y_test

# > Create model
def create_model():
    # Print info 
    print("[INFO] Initializing model")
    
    # > Initialize model 
    model = VGG16(include_top = False, # Do not include classifier!
                  pooling = "avg", # Pooling the final layer  
                  input_shape = (210, 210, 3)) # Defineing input shape
    # Disable training on convolutional layers
    for layer in model.layers:
        layer.trainable = False
        
    # > Add layers 
    # The second pair of closed brackets is the input 
    flat1 = Flatten()(model.layers[-1].output) # create a flatten layer from the output for the last layer of the model
    class1 = Dense(128, activation='relu')(flat1)
    output = Dense(28, activation='softmax')(class1)
    # Adding everything together
    model = Model(inputs = model.inputs, 
                  outputs = output)
    
    # Print info
    print("[INFO] Compiling model")
    # Slowing down the model's learning to avoid overfitting
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=1000,
        decay_rate=0.9)

    sgd = SGD(learning_rate=lr_schedule)
    # Compiling model
    model.compile(optimizer=sgd,
             loss="categorical_crossentropy", # binary_crossentropy for binary categories 
             metrics=["accuracy"])
    
    # Print info
    print("[INFO] Model compiled!")
    print("[INFO] Model summary:")
    model.summary()
    
    return model    

# > Evaluate model
def report(model, X_test, y_test):
    # Print info 
    print("[info] Reporting results...")
    
    # > Load label names
    # Get the filepath
    filepath = os.path.join("in", "processed_data", "label_names.csv")
    # Reading the filepath 
    label_names = pd.read_csv(filepath)
    
    # evaluate network
    predictions = model.predict(X_test, batch_size=32)
    # print classification report
    report = classification_report(y_test.argmax(axis=1), 
                                   predictions.argmax(axis=1), 
                                   target_names=label_names["labels"])
    # Print metrics
    print(report)
    # Save metrics
    outpath = os.path.join("output", "classification_report.txt")
    with open(outpath, "w") as f:
        f.write(report)
        
# > Plot history
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    # Saving image
    plt.savefig(os.path.join("output", "history_img.png"))

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
    X_train, X_test, y_train, y_test = load_data_nn(args["subset"])
    # Normalize data 
    X_train, X_test, y_train, y_test = normalize(X_train, X_test, y_train, y_test)
    # Create model
    model = create_model()
    
    # Train model
    history = model.fit(X_train, y_train,
             validation_data = (X_test, y_test), # Was there a way to split up the validation data further?
             batch_size = 128, # two to the power of something to optimize memory
             epochs = EPOCHS,
             validation_split = 0.1,
             verbose = 1) 
    # Report classification metrics 
    report(model, X_test, y_test)
    # Plot history
    plot_history(history, EPOCHS)
    
    # Print info 
    print("[INFO] Job complete")

# Run main() function from terminal only
if __name__ == "__main__":
    main()