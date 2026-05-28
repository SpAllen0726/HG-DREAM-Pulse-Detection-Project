#!/usr/bin/env python
# coding: utf-8

# In[12]:


import qkeras
from qkeras import QActivation
from qkeras import QDense, QConv1D
from tensorflow.keras.layers import Input, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l1
from qkeras.quantizers import quantized_bits, quantized_relu

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
import numpy as np
import os


X_data = np.load("X_Data_Bank.npy")
y_data = np.load("Y_Data_Bank.npy")
TIME_STEPS = np.shape(X_data)[1]
SAVE_DIR2 = '1Conv_checkpoints_running2'
LOG_FILE2 = '1Conv_3pulse_noise_tb232.csv'
MODEL_NAME_TEMPLATE2 = '1Conv_2pulse_noise.loss_{val_loss:01.5f}.e{epoch:03d}_deconv2.h5'
checkpoint_path = os.path.join(SAVE_DIR2, MODEL_NAME_TEMPLATE2)

VAL_SPLIT = 0.1

qmodel = Sequential([
    Input(shape=(TIME_STEPS, 1)),
    QActivation(activation=quantized_relu(6)),
    QConv1D(32, 3, padding='same'),
    QActivation(activation=quantized_relu(6)),
    QConv1D(32, 3, padding='same'),
    QActivation(activation=quantized_relu(6)),
    QConv1D(32, 3, padding='same'),
    QActivation(activation=quantized_relu(6)),
    QConv1D(32, 3, padding='same'),
    QActivation(activation=quantized_relu(6)),
    QConv1D(32, 3, padding='same'),
    QActivation(activation=quantized_relu(6)),
    QConv1D(64, 3, padding='same'),
    QActivation(activation=quantized_relu(6)),
    QConv1D(1, 9, padding='same'),
    QActivation(activation=quantized_relu(6)),
    Flatten(),
    QDense(TIME_STEPS, activation='softmax'), #, kernel_regularizer=l1(0.0001)
    #QActivation(quantized_bits(bits=24, integer=8)),
])


#pruning_params = {"pruning_schedule": pruning_schedule.ConstantSparsity(0.05, begin_step=2000, frequency=100)}
#qmodel = prune.prune_low_magnitude(qmodel, **pruning_params)

qmodel.summary()

# --- Compile Model ---
optimizer = Adam()
qmodel.compile(loss='mean_squared_logarithmic_error', optimizer=optimizer) 

# --- Setup Checkpoints & Callbacks ---
if not os.path.isdir(SAVE_DIR2):
    os.makedirs(SAVE_DIR2)

checkpoint_path2 = os.path.join(SAVE_DIR2, MODEL_NAME_TEMPLATE2)

callbacks = [
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
    EarlyStopping(monitor="val_loss", mode="min", patience=10),
    CSVLogger(LOG_FILE2, append=True, separator=',')
]

# --- Train Model ---
qmodel.fit(
    X_data, y_data,
    epochs=50,
    shuffle=True,
    validation_split=VAL_SPLIT,
    callbacks=callbacks
)

#qmodel = strip_pruning(qmodel)
qmodel.save("hgdream_new/quantized_model.h5")


# In[ ]:




