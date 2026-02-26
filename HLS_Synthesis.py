#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model

os.environ['PATH'] = '/opt/Xilinx/Vivado/2019.2/bin:' + os.environ['PATH']


# In[4]:


import hls4ml
import plotting


model = load_model('hgdream_new/unpruned_model.h5')

# First, the baseline model
hls_config = hls4ml.utils.config_from_keras_model(
    model, granularity='name', backend='Vivado', default_precision='ap_fixed<24,8>'
)

'''plotting.print_dict(hls_config)'''


hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=hls_config,
    backend='Vivado',
    output_dir='model_1/hls4ml_prj',
    part='xcu250-figd2104-2L-e',
    io_type='io_stream',
)
hls_model.compile()

