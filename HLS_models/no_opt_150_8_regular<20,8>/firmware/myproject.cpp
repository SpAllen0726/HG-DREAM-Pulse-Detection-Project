#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_5,
    hls::stream<result_t> &layer10_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_5,layer10_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<conv1d_17_weight_t, 24>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv1d_17_bias_t, 8>(b2, "b2.txt");
        nnet::load_weights_from_txt<conv1d_18_weight_t, 192>(w4, "w4.txt");
        nnet::load_weights_from_txt<conv1d_18_bias_t, 8>(b4, "b4.txt");
        nnet::load_weights_from_txt<conv1d_19_weight_t, 192>(w6, "w6.txt");
        nnet::load_weights_from_txt<conv1d_19_bias_t, 8>(b6, "b6.txt");
        nnet::load_weights_from_txt<conv1d_20_weight_t, 8>(w14, "w14.txt");
        nnet::load_weights_from_txt<conv1d_20_bias_t, 1>(b14, "b14.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=82
    nnet::zeropad1d_cl<input_t, layer11_t, config11>(input_5, layer11_out); // zp1d_conv1d_17

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=80
    nnet::conv_1d_cl<layer11_t, layer2_t, config2>(layer11_out, layer2_out, w2, b2); // conv1d_17

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=80
    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out); // conv1d_17_relu

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=82
    nnet::zeropad1d_cl<layer3_t, layer12_t, config12>(layer3_out, layer12_out); // zp1d_conv1d_18

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=80
    nnet::conv_1d_cl<layer12_t, layer4_t, config4>(layer12_out, layer4_out, w4, b4); // conv1d_18

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=82
    nnet::zeropad1d_cl<layer4_t, layer13_t, config13>(layer4_out, layer13_out); // zp1d_conv1d_19

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=80
    nnet::conv_1d_cl<layer13_t, layer6_t, config6>(layer13_out, layer6_out, w6, b6); // conv1d_19

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=80
    nnet::pointwise_conv_1d_cl<layer6_t, layer14_t, config14>(layer6_out, layer14_out, w14, b14); // conv1d_20

    nnet::sigmoid<layer14_t, result_t, sigmoid_config10>(layer14_out, layer10_out); // activation_7

}
