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
#define OUT_WIDTH_20 82
#define N_CHAN_20 1
#define N_OUTPUTS_2 80
#define N_FILT_2 32
#define N_OUTPUTS_2 80
#define N_FILT_2 32
#define OUT_WIDTH_21 82
#define N_CHAN_21 32
#define N_OUTPUTS_5 80
#define N_FILT_5 32
#define N_OUTPUTS_5 80
#define N_FILT_5 32
#define OUT_WIDTH_22 82
#define N_CHAN_22 32
#define N_OUTPUTS_8 80
#define N_FILT_8 32
#define N_OUTPUTS_8 80
#define N_FILT_8 32
#define OUT_WIDTH_23 82
#define N_CHAN_23 32
#define N_OUTPUTS_11 80
#define N_FILT_11 32
#define N_OUTPUTS_11 80
#define N_FILT_11 32
#define OUT_WIDTH_24 82
#define N_CHAN_24 32
#define N_OUTPUTS_14 80
#define N_FILT_14 32
#define N_OUTPUTS_14 80
#define N_FILT_14 32
#define N_OUTPUTS_25 80
#define N_FILT_25 1
#define N_OUTPUTS_17 80
#define N_FILT_17 1

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<14,8>, 1*1> input_t;
typedef nnet::array<ap_fixed<14,8>, 1*1> layer20_t;
typedef ap_fixed<14,8> q_conv1d_17_accum_t;
typedef nnet::array<ap_fixed<14,8>, 32*1> layer2_t;
typedef ap_fixed<8,3> weight2_t;
typedef ap_fixed<8,3> bias2_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer4_t;
typedef ap_fixed<18,8> q_activation_17_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer21_t;
typedef ap_fixed<14,8> q_conv1d_18_accum_t;
typedef nnet::array<ap_fixed<14,8>, 32*1> layer5_t;
typedef ap_fixed<8,3> weight5_t;
typedef ap_fixed<8,3> bias5_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer7_t;
typedef ap_fixed<18,8> q_activation_18_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer22_t;
typedef ap_fixed<14,8> q_conv1d_19_accum_t;
typedef nnet::array<ap_fixed<14,8>, 32*1> layer8_t;
typedef ap_fixed<8,3> weight8_t;
typedef ap_fixed<8,3> bias8_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer10_t;
typedef ap_fixed<18,8> q_activation_19_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer23_t;
typedef ap_fixed<14,8> q_conv1d_20_accum_t;
typedef nnet::array<ap_fixed<14,8>, 32*1> layer11_t;
typedef ap_fixed<8,3> weight11_t;
typedef ap_fixed<8,3> bias11_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer13_t;
typedef ap_fixed<18,8> q_activation_20_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer24_t;
typedef ap_fixed<14,8> q_conv1d_21_accum_t;
typedef nnet::array<ap_fixed<14,8>, 32*1> layer14_t;
typedef ap_fixed<8,3> weight14_t;
typedef ap_fixed<8,3> bias14_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer16_t;
typedef ap_fixed<18,8> q_activation_21_table_t;
typedef ap_fixed<14,8> q_conv1d_22_accum_t;
typedef nnet::array<ap_fixed<14,8>, 1*1> layer25_t;
typedef ap_fixed<8,3> weight25_t;
typedef ap_fixed<8,3> bias25_t;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT>, 1*1> result_t;
typedef ap_ufixed<2,0> slope19_t;
typedef ap_ufixed<2,0> shift19_t;
typedef ap_fixed<18,8> q_activation_22_table_t;

#endif
