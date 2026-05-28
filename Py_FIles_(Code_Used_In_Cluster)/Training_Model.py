#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import os
import tensorflow.keras.models 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam

INPUT_FILE = 'hgdream/Sensl_FastOut_AveragePulse_1p8GHzBandwidth.feather'
SAVE_DIR = '1Conv_checkpoints_running'
LOG_FILE = '1Conv_3pulse_noise_tb23.csv'
SAVE_DIR2 = '1Conv_checkpoints_running2'
LOG_FILE2 = '1Conv_3pulse_noise_tb232.csv'
MODEL_NAME_TEMPLATE = '1Conv_2pulse_noise.loss_{val_loss:01.5f}.e{epoch:03d}_deconv.h5'
MODEL_NAME_TEMPLATE2 = '1Conv_2pulse_noise.loss_{val_loss:01.5f}.e{epoch:03d}_deconv2.h5'
NUM_EVENTS = 500
TIME_STEPS = 80 # divides width_cutoff
width_cutoff = 800
SCALE_FACTOR = 12.0 / 10.0
VAL_SPLIT = 0.1


X_data = np.load("X_Data_Bank.npy")
y_data = np.load("Y_Data_Bank.npy")

# --- Build Model ---
Train = True
if Train:
    model = Sequential([
        Input(shape=(TIME_STEPS, 1)),
        Conv1D(64, 3, activation='relu', padding='same'),
        Conv1D(32, 3, activation='relu', padding='same'),
        *[Conv1D(32, 3, activation='relu', padding='same') for _ in range(6)],
        Conv1D(64, 3, activation='relu', padding='same'),
        Conv1D(1, 9, activation='relu', padding='same'),
        Flatten(),
        Dense(TIME_STEPS, activation='relu')
    ])

    model.summary()

    # --- Compile Model ---
    optimizer = Adam(amsgrad=True)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=optimizer) #mean_squared_logarithmic_error

    # --- Setup Checkpoints & Callbacks ---
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    checkpoint_path = os.path.join(SAVE_DIR, MODEL_NAME_TEMPLATE)

    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", mode="min", patience=10),
        CSVLogger(LOG_FILE, append=True, separator=',')
    ]

    # --- Train Model ---
    model.fit(
        X_data, y_data,
        epochs=30,
        shuffle=True,
        validation_split=VAL_SPLIT,
        callbacks=callbacks
    )

    model.save('unpruned_model.h5')
else:
    from tensorflow.keras.models import load_model

    model = load_model('unpruned_model.h5')

