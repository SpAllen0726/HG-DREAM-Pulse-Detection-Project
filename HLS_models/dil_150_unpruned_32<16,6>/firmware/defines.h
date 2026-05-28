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
#define OUT_WIDTH_17 82
#define N_CHAN_17 1
#define N_OUTPUTS_2 80
#define N_FILT_2 32
#define N_OUTPUTS_2 80
#define N_FILT_2 32
#define OUT_WIDTH_18 82
#define N_CHAN_18 32
#define N_OUTPUTS_5 80
#define N_FILT_5 32
#define N_OUTPUTS_5 80
#define N_FILT_5 32
#define OUT_WIDTH_19 82
#define N_CHAN_19 32
#define N_OUTPUTS_8 80
#define N_FILT_8 32
#define N_OUTPUTS_8 80
#define N_FILT_8 32
#define OUT_WIDTH_20 82
#define N_CHAN_20 32
#define N_OUTPUTS_11 80
#define N_FILT_11 32
#define N_OUTPUTS_11 80
#define N_FILT_11 32
#define N_OUTPUTS_21 80
#define N_FILT_21 1
#define N_OUTPUTS_14 80
#define N_FILT_14 1

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,6>, 1*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer17_t;
typedef ap_fixed<16,6> q_conv1d_5_accum_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer2_t;
typedef ap_fixed<8,3> weight2_t;
typedef ap_fixed<8,3> bias2_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer4_t;
typedef ap_fixed<18,8> q_activation_5_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer18_t;
typedef ap_fixed<16,6> q_conv1d_6_accum_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer5_t;
typedef ap_fixed<8,3> weight5_t;
typedef ap_fixed<8,3> bias5_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer7_t;
typedef ap_fixed<18,8> q_activation_6_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer19_t;
typedef ap_fixed<16,6> q_conv1d_7_accum_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer8_t;
typedef ap_fixed<8,3> weight8_t;
typedef ap_fixed<8,3> bias8_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer10_t;
typedef ap_fixed<18,8> q_activation_7_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer20_t;
typedef ap_fixed<16,6> q_conv1d_8_accum_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer11_t;
typedef ap_fixed<8,3> weight11_t;
typedef ap_fixed<8,3> bias11_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 32*1> layer13_t;
typedef ap_fixed<18,8> q_activation_8_table_t;
typedef ap_fixed<16,6> q_conv1d_9_accum_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer21_t;
typedef ap_fixed<8,3> weight21_t;
typedef ap_fixed<8,3> bias21_t;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT>, 1*1> result_t;
typedef ap_ufixed<2,0> slope16_t;
typedef ap_ufixed<2,0> shift16_t;
typedef ap_fixed<18,8> q_activation_9_table_t;

#endif
