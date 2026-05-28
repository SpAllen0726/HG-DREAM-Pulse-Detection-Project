#!/usr/bin/env python
# coding: utf-8

# In[5]:


from tensorflow.keras.models import load_model
import hls4ml
from qkeras.utils import _add_supported_quantized_objects


# In[7]:


resolution = 150


# In[8]:


co = {}
_add_supported_quantized_objects(co)
qmodel = load_model(f"quantized_model_opt_{resolution}_pruning.h5", custom_objects=co)
hls_config_q = hls4ml.utils.config_from_keras_model(qmodel, granularity='name', backend='Vitis')


# In[9]:


synthesize = True
if synthesize:
    hls_model_q = hls4ml.converters.convert_from_keras_model(
        qmodel, hls_config=hls_config_q, output_dir='quantized_cnn_final', backend='Vitis', io_type='io_stream'
    )
    hls_model_q.compile()
    hls_model_q.build(csim=False, synth=True, vsynth=True)


# In[ ]:


'''def getReports(indir):
    data_ = {}

    report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
    report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))

    if report_vsynth.is_file() and report_csynth.is_file():
        print('Found valid vsynth and synth in {}! Fetching numbers'.format(indir))

        # Get the resources from the logic synthesis report
        with report_vsynth.open() as report:
            lines = np.array(report.readlines())
            data_['lut'] = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
            data_['ff'] = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
            data_['bram'] = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
            data_['dsp'] = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
            data_['lut_rel'] = float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5])
            data_['ff_rel'] = float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5])
            data_['bram_rel'] = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5])
            data_['dsp_rel'] = float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5])

        with report_csynth.open() as report:
            lines = np.array(report.readlines())
            lat_line = lines[np.argwhere(np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
            data_['latency_clks'] = int(lat_line.split('|')[2])
            data_['latency_mus'] = float(lat_line.split('|')[2]) * 5.0 / 1000.0
            data_['latency_ii'] = int(lat_line.split('|')[6])

    return data_'''


# In[ ]:


'''
from pathlib import Path
import pprint
if synthesize:
    data_quantized_pruned = getReports('quantized_cnn_final')
    
    print("\n Resource usage and latency: Quantized")
    pprint.pprint(data_quantized_pruned)'''

