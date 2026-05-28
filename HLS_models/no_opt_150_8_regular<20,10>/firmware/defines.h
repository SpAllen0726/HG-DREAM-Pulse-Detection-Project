#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 80
#define N_INPUT_2_1 1
#define OUT_WIDTH_11 82
#define N_CHAN_11 1
#define N_OUTPUTS_2 80
#define N_FILT_2 8
#define N_OUTPUTS_2 80
#define N_FILT_2 8
#define OUT_WIDTH_12 82
#define N_CHAN_12 8
#define N_OUTPUTS_4 80
#define N_FILT_4 8
#define OUT_WIDTH_13 82
#define N_CHAN_13 8
#define N_OUTPUTS_6 80
#define N_FILT_6 8
#define N_OUTPUTS_14 80
#define N_FILT_14 1
#define N_OUTPUTS_8 80
#define N_FILT_8 1

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<20,10>, 1*1> input_t;
typedef nnet::array<ap_fixed<20,10>, 1*1> layer11_t;
typedef ap_fixed<20,10> conv1d_17_accum_t;
typedef nnet::array<ap_fixed<20,10>, 8*1> layer2_t;
typedef ap_fixed<20,10> conv1d_17_weight_t;
typedef ap_fixed<20,10> conv1d_17_bias_t;
typedef nnet::array<ap_fixed<20,10>, 8*1> layer3_t;
typedef ap_fixed<18,8> conv1d_17_relu_table_t;
typedef nnet::array<ap_fixed<20,10>, 8*1> layer12_t;
typedef ap_fixed<20,10> conv1d_18_accum_t;
typedef nnet::array<ap_fixed<20,10>, 8*1> layer4_t;
typedef ap_fixed<20,10> conv1d_18_weight_t;
typedef ap_fixed<20,10> conv1d_18_bias_t;
typedef nnet::array<ap_fixed<20,10>, 8*1> layer13_t;
typedef ap_fixed<20,10> conv1d_19_accum_t;
typedef nnet::array<ap_fixed<20,10>, 8*1> layer6_t;
typedef ap_fixed<20,10> conv1d_19_weight_t;
typedef ap_fixed<20,10> conv1d_19_bias_t;
typedef ap_fixed<20,10> conv1d_20_accum_t;
typedef nnet::array<ap_fixed<20,10>, 1*1> layer14_t;
typedef ap_fixed<20,10> conv1d_20_weight_t;
typedef ap_fixed<20,10> conv1d_20_bias_t;
typedef nnet::array<ap_fixed<20,10>, 1*1> result_t;
typedef ap_fixed<18,8> activation_7_table_t;

#endif
