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
#define OUT_WIDTH_18 82
#define N_CHAN_18 1
#define N_OUTPUTS_2 80
#define N_FILT_2 32
#define N_OUTPUTS_2 80
#define N_FILT_2 32
#define OUT_WIDTH_19 81
#define N_CHAN_19 32
#define N_OUTPUTS_5 40
#define N_FILT_5 32
#define N_OUTPUTS_5 40
#define N_FILT_5 32
#define OUT_WIDTH_20 42
#define N_CHAN_20 32
#define N_OUTPUTS_8 40
#define N_FILT_8 64
#define N_OUTPUTS_8 40
#define N_FILT_8 64
#define OUT_WIDTH_11 80
#define N_CHAN_11 64
#define OUT_WIDTH_21 82
#define N_CHAN_21 64
#define N_OUTPUTS_12 80
#define N_FILT_12 32
#define N_OUTPUTS_12 80
#define N_FILT_12 32
#define N_OUTPUTS_22 80
#define N_FILT_22 1
#define N_OUTPUTS_15 80
#define N_FILT_15 1

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,8>, 1*1> input_t;
typedef nnet::array<ap_fixed<16,8>, 1*1> layer18_t;
typedef ap_fixed<16,8> q_conv1d_accum_t;
typedef nnet::array<ap_fixed<16,8>, 32*1> layer2_t;
typedef ap_fixed<8,3> weight2_t;
typedef ap_fixed<8,3> bias2_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer4_t;
typedef ap_fixed<18,8> q_activation_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer19_t;
typedef ap_fixed<16,8> q_conv1d_1_accum_t;
typedef nnet::array<ap_fixed<16,8>, 32*1> layer5_t;
typedef ap_fixed<8,3> weight5_t;
typedef ap_fixed<8,3> bias5_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer7_t;
typedef ap_fixed<18,8> q_activation_1_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer20_t;
typedef ap_fixed<16,8> q_conv1d_2_accum_t;
typedef nnet::array<ap_fixed<16,8>, 64*1> layer8_t;
typedef ap_fixed<8,3> weight8_t;
typedef ap_fixed<8,3> bias8_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 64*1> layer10_t;
typedef ap_fixed<18,8> q_activation_2_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 64*1> layer11_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 64*1> layer21_t;
typedef ap_fixed<16,8> q_conv1d_3_accum_t;
typedef nnet::array<ap_fixed<16,8>, 32*1> layer12_t;
typedef ap_fixed<8,3> weight12_t;
typedef ap_fixed<8,3> bias12_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer14_t;
typedef ap_fixed<18,8> q_activation_3_table_t;
typedef ap_fixed<16,8> q_conv1d_4_accum_t;
typedef nnet::array<ap_fixed<16,8>, 1*1> layer22_t;
typedef ap_fixed<8,3> weight22_t;
typedef ap_fixed<8,3> bias22_t;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT>, 1*1> result_t;
typedef ap_ufixed<2,0> slope17_t;
typedef ap_ufixed<2,0> shift17_t;
typedef ap_fixed<18,8> q_activation_4_table_t;

#endif
