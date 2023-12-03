#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow h5py numpy sklearn


# In[2]:


pip install numpy h5py tensorflow scikit-learn


# In[3]:


pip install pyaudio


# In[4]:


import h5py
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import math
import pyaudio
import tensorflow as tf


# In[5]:


# Load HDF5 file
file_path = 'C:/Users/anush/Downloads/noxi_dataset (2).hdf5'
hdf5_file = h5py.File(file_path, 'r')

# Since we don't have a specific utt-id, we'll take a look at the structure of the first key in the HDF5 file
first_key = list(hdf5_file.keys())[0]

# Extract the data for the first utterance
utt_data = hdf5_file[first_key]
fbank = utt_data['fbank'][:]
pitch = utt_data['pitch'][:]
pitch_confidence = utt_data['pitch_confidence'][:]
vad = utt_data['vad'][:]
labels = utt_data['labels'][:]

# Close the HDF5 file
hdf5_file.close()

# Plotting the last few frames of the utterance
# We will plot the fbank, pitch, pitch confidence, and labels for the last 50 frames
n_frames_to_plot = 50

# Define a function to plot the data
def plot_data(data, title, xlabel, ylabel, start_frame):
    plt.figure(figsize=(10, 2))
    plt.plot(data[-n_frames_to_plot:], label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(start_frame, start_frame + n_frames_to_plot - 1)
    plt.legend()
    plt.show()

# Plot the features
plot_data(fbank, 'Fbank', 'Frames', 'Feature Value', fbank.shape[0] - n_frames_to_plot)
plot_data(pitch, 'Pitch', 'Frames', 'Pitch Value', pitch.shape[0] - n_frames_to_plot)
plot_data(pitch_confidence, 'Pitch Confidence', 'Frames', 'Confidence Value', pitch_confidence.shape[0] - n_frames_to_plot)
plot_data(vad, 'VAD Probability', 'Frames', 'Probability', vad.shape[0] - n_frames_to_plot)
plot_data(labels, 'Labels', 'Frames', 'Label', labels.shape[0] - n_frames_to_plot)


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Since we can't process the entire dataset at once due to its potential size, we'll make a function
# that yields processed segments of the data for training.
def data_generator(hdf5_path, batch_size=64, num_frames=10):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        # Get all utterance IDs
        all_keys = list(hdf5_file.keys())
        while True:  # Loop forever so the generator never terminates
            # Shuffle utterance IDs to prevent model from learning any potential order
            np.random.shuffle(all_keys)

            for start_idx in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[start_idx:start_idx + batch_size]
                batch_data = []
                batch_labels = []
                for key in batch_keys:
                    # Extract data for the current utterance
                    utt_data = hdf5_file[key]
                    fbank = utt_data['fbank'][:]
                    pitch = utt_data['pitch'][:]
                    vad = utt_data['vad'][:]
                    labels = utt_data['labels'][:]

                    # Standardize features
                    scaler = StandardScaler()
                    fbank_norm = scaler.fit_transform(fbank)
                    pitch_norm = scaler.fit_transform(pitch.reshape(-1, 1)).flatten()
                    vad_norm = vad  # VAD is already a probability, so we may not need to normalize

                    # Create sequences of frames
                    for i in range(num_frames, len(fbank)):
                        features = np.hstack((fbank_norm[i-num_frames:i].flatten(),
                                              pitch_norm[i-num_frames:i],
                                              vad_norm[i-num_frames:i]))
                        batch_data.append(features)
                        batch_labels.append(labels[i])

                # Convert to arrays
                batch_data = np.array(batch_data)
                batch_labels = np.array(batch_labels)

                yield batch_data, to_categorical(batch_labels)

# Model definition function
def build_model(input_shape, output_shape):
    model = Sequential([
        Dense(256, input_shape=input_shape, activation='relu'),
        Dropout(0.5),
             Dense(output_shape, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

# Since we don't know the total number of frames we'll encounter,
# we will make a reasonable assumption for demonstration purposes.
# For the actual model, you should calculate the actual number of frames you have.
# Let's assume we have 10000 frames.
total_frames = 12000
num_frames = 10  # Number of frames to look back
batch_size = 64  # Size of the batch
steps_per_epoch = total_frames // batch_size  # Steps per epoch for the generator

# Prepare the data generator
generator = data_generator(file_path, batch_size=batch_size, num_frames=num_frames)

# Build the model with the appropriate input shape
# The input shape will be the size of the feature vector for each frame
# multiplied by the number of frames we look back.
input_shape = (num_frames * (80 + 1 + 1),)  # 80 fbank + 1 pitch + 1 VAD
output_shape = 2  # Two classes: 0 or 1

# Creating a model instance
model = build_model(input_shape, output_shape)

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')

# Fit the model using the data generator
# Note: In an actual scenario, you should have a separate validation generator
# to monitor the validation loss for early stopping.
# Here, we'll just use the training generator as a placeholder.
history = model.fit(generator, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=5,  # For demonstration, we'll use just a few epochs.
                    callbacks=[early_stopping])

# Normally, you would also include validation data to monitor for overfitting.
# However, for this demonstration, we will omit that step.

# Let's return the model and the training history for inspection
model, history.history



# In[ ]:





# In[7]:


import numpy as np
import h5py  # Import h5py library
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Data generator function
def data_generator(hdf5_path, batch_size=64, num_frames=10):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        all_keys = list(hdf5_file.keys())
        while True:
            np.random.shuffle(all_keys)

            for start_idx in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[start_idx:start_idx + batch_size]
                batch_data = []
                batch_labels = []
                for key in batch_keys:
                    utt_data = hdf5_file[key]
                    fbank = utt_data['fbank'][:]
                    pitch = utt_data['pitch'][:]
                    vad = utt_data['vad'][:]
                    labels = utt_data['labels'][:]

                    scaler = StandardScaler()
                    fbank_norm = scaler.fit_transform(fbank)
                    pitch_norm = scaler.fit_transform(pitch.reshape(-1, 1)).flatten()
                    vad_norm = vad

                    for i in range(num_frames, len(fbank)):
                        features = np.hstack((fbank_norm[i-num_frames:i],
                                              pitch_norm[i-num_frames:i].reshape(-1, 1),
                                              vad_norm[i-num_frames:i].reshape(-1, 1)))
                        batch_data.append(features.reshape(num_frames, -1))
                        batch_labels.append(labels[i])

                batch_data = np.array(batch_data)
                batch_labels = np.array(batch_labels)

                yield batch_data, to_categorical(batch_labels)

# LSTM model definition
def build_lstm_model(input_shape, output_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.5),
        Dense(output_shape, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

# Define file_path to your HDF5 data file
file_path = 'C:/Users/anush/Downloads/noxi_dataset (2).hdf5'  # Replace with your actual file path

# Training parameters
total_frames = 12000  # Adjust based on your dataset
num_frames = 10  # Number of frames to look back
batch_size = 64  # Size of the batch
steps_per_epoch = total_frames // batch_size

# Input shape and output shape
input_shape = (num_frames, 82)  # num_frames time steps, 82 features
output_shape = 2  # Number of output classes

# Create the data generator
generator = data_generator(file_path, batch_size=batch_size, num_frames=num_frames)

# Creating the LSTM model
lstm_model = build_lstm_model(input_shape, output_shape)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')

# Fit the LSTM model  
history = lstm_model.fit(generator, 
                         steps_per_epoch=steps_per_epoch, 
                         epochs=5,  # Number of epochs
                         callbacks=[early_stopping])


# In[8]:


lstm_model.save('C:/Users/anush/Downloads/my_lstm_model.keras')


# In[9]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

# Assuming you have a validation dataset in a similar HDF5 format
validation_file_path = 'C:/Users/anush/OneDrive/Desktop/Noxi_dataset/zoom_dataset.hdf5'  # Replace with your actual validation file path

# Create the data generator for the validation data
validation_generator = data_generator(validation_file_path, batch_size=batch_size, num_frames=num_frames)

# Number of steps for validation
# Adjust total_validation_frames based on your validation dataset size
total_validation_frames = 6000  # Example number
validation_steps = total_validation_frames // batch_size

# Predict on the validation data
predictions = []
actuals = []
for _ in range(validation_steps):
    x, y = next(validation_generator)
    y_pred = lstm_model.predict(x)
    predictions.extend(y_pred)
    actuals.extend(y)

# Convert predictions and actuals to numpy arrays
predictions = np.array(predictions)
actuals = np.array(actuals)

# Calculate MAE and RMSE
# Assuming your task is a regression. If it's a classification, adjust accordingly
mae = mean_absolute_error(actuals, predictions)
rmse = math.sqrt(mean_squared_error(actuals, predictions))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Square Error (RMSE):", rmse)


# In[10]:


import h5py

# Path to your HDF5 file
file_path = 'C:/Users/anush/Downloads/noxi_dataset (2).hdf5'

# Open the HDF5 file and list all top-level groups/datasets
with h5py.File(file_path, 'r') as hdf:
    print("Datasets/Groups within the HDF5 file:")
    for key in hdf.keys():
        print(key)


# In[11]:


import h5py

# Replace with the path to your HDF5 file
file_path = 'C:/Users/anush/Downloads/noxi_dataset (2).hdf5'

# Function to recursively print the structure of the HDF5 file
def print_structure(name, obj):
    print(name, 'Group' if isinstance(obj, h5py.Group) else 'Dataset')

# Open the HDF5 file
with h5py.File(file_path, 'r') as hdf:
    # Print the structure of the file
    print("Structure of the HDF5 file:")
    hdf.visititems(print_structure)

    # After printing the structure, identify the path to your dataset
    # Replace 'path/to/your/dataset' with the actual path
    dataset_path = 'C:/Users/anush/Downloads/noxi_dataset (2).hdf5/nox_n8166_035'

    # Check if the dataset exists in the file
    if dataset_path in hdf:
        # Access the dataset
        dataset = hdf[dataset_path]

        # Assuming the dataset is a numpy array or similar
        # Adjust the slicing as needed
        sample_data = dataset[:100]  # Load the first 100 entries
        print(f"\nShape of the sample data from {dataset_path}: {sample_data.shape}")
        print("Sample data:", sample_data)
    else:
        print(f"\nDataset {dataset_path} not found in the HDF5 file.")

# Note: Replace 'path/to/your/dataset' with the actual dataset path


# In[ ]:





# In[ ]:


import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Optimized LSTM model definition
def build_optimized_lstm_model(input_shape, output_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        Dropout(0.5),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(output_shape, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0005),  # Adjust learning rate as needed
                  metrics=['accuracy'])
    return model

# Data generator function
def data_generator(hdf5_path, batch_size=64, num_frames=10):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        all_keys = list(hdf5_file.keys())
        while True:
            np.random.shuffle(all_keys)

            for start_idx in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[start_idx:start_idx + batch_size]
                batch_data = []
                batch_labels = []
                for key in batch_keys:
                    utt_data = hdf5_file[key]
                    fbank = utt_data['fbank'][:]
                    pitch = utt_data['pitch'][:]
                    vad = utt_data['vad'][:]
                    labels = utt_data['labels'][:]

                    scaler = StandardScaler()
                    fbank_norm = scaler.fit_transform(fbank)
                    pitch_norm = scaler.fit_transform(pitch.reshape(-1, 1)).flatten()
                    vad_norm = vad

                    for i in range(num_frames, len(fbank)):
                        features = np.hstack((fbank_norm[i-num_frames:i],
                                              pitch_norm[i-num_frames:i].reshape(-1, 1),
                                              vad_norm[i-num_frames:i].reshape(-1, 1)))
                        batch_data.append(features.reshape(num_frames, -1))
                        batch_labels.append(labels[i])

                batch_data = np.array(batch_data)
                batch_labels = np.array(batch_labels)

                yield batch_data, to_categorical(batch_labels)

# Define file_path to your HDF5 data file
file_path = 'C:/Users/anush/Downloads/noxi_dataset (2).hdf5'  # Replace with your actual file path

# Training parameters
total_frames = 12000  # Adjust based on your dataset
num_frames = 10  # Number of frames to look back
batch_size = 64  # Size of the batch
steps_per_epoch = total_frames // batch_size

# Input shape and output shape
input_shape = (num_frames, 82)  # num_frames time steps, 82 features
output_shape = 2  # Number of output classes

# Create the data generator
generator = data_generator(file_path, batch_size=batch_size, num_frames=num_frames)

# Creating the optimized LSTM model
lstm_model = build_optimized_lstm_model(input_shape, output_shape)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
r=[]
# Fit the LSTM model
history = lstm_model.fit(generator, 
                         steps_per_epoch=steps_per_epoch, 
                         epochs=5,  # Number of epochs
                         callbacks=[early_stopping])


# In[ ]:




