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
#define OUT_WIDTH_14 82
#define N_CHAN_14 1
#define N_OUTPUTS_2 80
#define N_FILT_2 32
#define N_OUTPUTS_2 80
#define N_FILT_2 32
#define OUT_WIDTH_15 82
#define N_CHAN_15 32
#define N_OUTPUTS_5 80
#define N_FILT_5 32
#define N_OUTPUTS_5 80
#define N_FILT_5 32
#define OUT_WIDTH_16 82
#define N_CHAN_16 32
#define N_OUTPUTS_8 80
#define N_FILT_8 32
#define N_OUTPUTS_8 80
#define N_FILT_8 32
#define N_OUTPUTS_17 80
#define N_FILT_17 1
#define N_OUTPUTS_11 80
#define N_FILT_11 1

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<20,2>, 1*1> input_t;
typedef nnet::array<ap_fixed<20,2>, 1*1> layer14_t;
typedef ap_fixed<20,2> q_conv1d_24_accum_t;
typedef nnet::array<ap_fixed<20,2>, 32*1> layer2_t;
typedef ap_fixed<8,3> weight2_t;
typedef ap_fixed<8,3> bias2_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer4_t;
typedef ap_fixed<18,8> q_activation_24_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer15_t;
typedef ap_fixed<20,2> q_conv1d_25_accum_t;
typedef nnet::array<ap_fixed<20,2>, 32*1> layer5_t;
typedef ap_fixed<8,3> weight5_t;
typedef ap_fixed<8,3> bias5_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer7_t;
typedef ap_fixed<18,8> q_activation_25_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer16_t;
typedef ap_fixed<20,2> q_conv1d_26_accum_t;
typedef nnet::array<ap_fixed<20,2>, 32*1> layer8_t;
typedef ap_fixed<8,3> weight8_t;
typedef ap_fixed<8,3> bias8_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer10_t;
typedef ap_fixed<18,8> q_activation_26_table_t;
typedef ap_fixed<20,2> q_conv1d_27_accum_t;
typedef nnet::array<ap_fixed<20,2>, 1*1> layer17_t;
typedef ap_fixed<8,3> weight17_t;
typedef ap_fixed<8,3> bias17_t;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT>, 1*1> result_t;
typedef ap_ufixed<2,0> slope13_t;
typedef ap_ufixed<2,0> shift13_t;
typedef ap_fixed<18,8> q_activation_27_table_t;

#endif
