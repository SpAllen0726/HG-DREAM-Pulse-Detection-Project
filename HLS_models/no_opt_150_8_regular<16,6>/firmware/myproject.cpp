#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_4,
    hls::stream<result_t> &layer13_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_4,layer13_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<conv1d_13_weight_t, 24>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv1d_13_bias_t, 8>(b2, "b2.txt");
        nnet::load_weights_from_txt<conv1d_14_weight_t, 192>(w5, "w5.txt");
        nnet::load_weights_from_txt<conv1d_14_bias_t, 8>(b5, "b5.txt");
        nnet::load_weights_from_txt<conv1d_15_weight_t, 192>(w8, "w8.txt");
        nnet::load_weights_from_txt<conv1d_15_bias_t, 8>(b8, "b8.txt");
        nnet::load_weights_from_txt<conv1d_16_weight_t, 8>(w17, "w17.txt");
        nnet::load_weights_from_txt<conv1d_16_bias_t, 1>(b17, "b17.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=82
    nnet::zeropad1d_cl<input_t, layer14_t, config14>(input_4, layer14_out); // zp1d_conv1d_13

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=80
    nnet::conv_1d_cl<layer14_t, layer2_t, config2>(layer14_out, layer2_out, w2, b2); // conv1d_13

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=80
    nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out); // activation_3

    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=82
    nnet::zeropad1d_cl<layer4_t, layer15_t, config15>(layer4_out, layer15_out); // zp1d_conv1d_14

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=80
    nnet::conv_1d_cl<layer15_t, layer5_t, config5>(layer15_out, layer5_out, w5, b5); // conv1d_14

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=80
    nnet::relu<layer5_t, layer7_t, relu_config7>(layer5_out, layer7_out); // activation_4

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=82
    nnet::zeropad1d_cl<layer7_t, layer16_t, config16>(layer7_out, layer16_out); // zp1d_conv1d_15

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=80
    nnet::conv_1d_cl<layer16_t, layer8_t, config8>(layer16_out, layer8_out, w8, b8); // conv1d_15

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=80
    nnet::relu<layer8_t, layer10_t, relu_config10>(layer8_out, layer10_out); // activation_5

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=80
    nnet::pointwise_conv_1d_cl<layer10_t, layer17_t, config17>(layer10_out, layer17_out, w17, b17); // conv1d_16

    nnet::sigmoid<layer17_t, result_t, sigmoid_config13>(layer17_out, layer13_out); // activation_6

}
