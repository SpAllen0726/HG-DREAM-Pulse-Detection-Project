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
#define OUT_WIDTH_2 82
#define N_CHAN_2 1
#define N_OUTPUTS_3 80
#define N_FILT_3 16
#define N_OUTPUTS_3 80
#define N_FILT_3 16
#define OUT_WIDTH_6 84
#define N_CHAN_6 16
#define N_OUTPUTS_7 82
#define N_FILT_7 16
#define N_OUTPUTS_7 82
#define N_FILT_7 16
#define OUT_WIDTH_10 90
#define N_CHAN_10 16
#define N_OUTPUTS_11 88
#define N_FILT_11 16
#define N_OUTPUTS_11 88
#define N_FILT_11 16
#define OUT_WIDTH_14 104
#define N_CHAN_14 16
#define N_OUTPUTS_15 102
#define N_FILT_15 16
#define N_OUTPUTS_15 102
#define N_FILT_15 16
#define N_OUTPUTS_21 102
#define N_FILT_21 1
#define N_OUTPUTS_18 102
#define N_FILT_18 1

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<40,20>, 1*1> input_t;
typedef nnet::array<ap_fixed<40,20>, 1*1> layer2_t;
typedef ap_fixed<40,20> q_conv1d_65_accum_t;
typedef nnet::array<ap_fixed<40,20>, 16*1> layer3_t;
typedef ap_fixed<8,3> weight3_t;
typedef ap_fixed<8,3> bias3_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 16*1> layer5_t;
typedef ap_fixed<18,8> q_activation_65_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 16*1> layer6_t;
typedef ap_fixed<40,20> q_conv1d_66_accum_t;
typedef nnet::array<ap_fixed<40,20>, 16*1> layer7_t;
typedef ap_fixed<8,3> weight7_t;
typedef ap_fixed<8,3> bias7_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 16*1> layer9_t;
typedef ap_fixed<18,8> q_activation_66_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 16*1> layer10_t;
typedef ap_fixed<40,20> q_conv1d_67_accum_t;
typedef nnet::array<ap_fixed<40,20>, 16*1> layer11_t;
typedef ap_fixed<8,3> weight11_t;
typedef ap_fixed<8,3> bias11_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 16*1> layer13_t;
typedef ap_fixed<18,8> q_activation_67_table_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 16*1> layer14_t;
typedef ap_fixed<40,20> q_conv1d_68_accum_t;
typedef nnet::array<ap_fixed<40,20>, 16*1> layer15_t;
typedef ap_fixed<8,3> weight15_t;
typedef ap_fixed<8,3> bias15_t;
typedef nnet::array<ap_ufixed<6,0,AP_RND_CONV,AP_SAT>, 16*1> layer17_t;
typedef ap_fixed<18,8> q_activation_68_table_t;
typedef ap_fixed<40,20> q_conv1d_69_accum_t;
typedef nnet::array<ap_fixed<40,20>, 1*1> layer21_t;
typedef ap_fixed<8,3> weight21_t;
typedef ap_fixed<8,3> bias21_t;
typedef nnet::array<ap_ufixed<8,0,AP_RND_CONV,AP_SAT>, 1*1> result_t;
typedef ap_ufixed<2,0> slope20_t;
typedef ap_ufixed<2,0> shift20_t;
typedef ap_fixed<18,8> q_activation_69_table_t;

#endif
