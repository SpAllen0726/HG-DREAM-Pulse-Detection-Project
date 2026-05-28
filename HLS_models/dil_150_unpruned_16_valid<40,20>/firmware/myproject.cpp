#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_14,
    hls::stream<result_t> &layer20_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_14,layer20_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight3_t, 48>(w3, "w3.txt");
        nnet::load_weights_from_txt<bias3_t, 16>(b3, "b3.txt");
        nnet::load_weights_from_txt<weight7_t, 768>(w7, "w7.txt");
        nnet::load_weights_from_txt<bias7_t, 16>(b7, "b7.txt");
        nnet::load_weights_from_txt<weight11_t, 768>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 16>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight15_t, 768>(w15, "w15.txt");
        nnet::load_weights_from_txt<bias15_t, 16>(b15, "b15.txt");
        nnet::load_weights_from_txt<weight21_t, 16>(w21, "w21.txt");
        nnet::load_weights_from_txt<bias21_t, 1>(b21, "b21.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=82
    nnet::zeropad1d_cl<input_t, layer2_t, config2>(input_14, layer2_out); // zero_padding1d_22

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=80
    nnet::conv_1d_cl<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3); // q_conv1d_65

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=80
    nnet::relu<layer3_t, layer5_t, relu_config5>(layer3_out, layer5_out); // q_activation_65

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=84
    nnet::zeropad1d_cl<layer5_t, layer6_t, config6>(layer5_out, layer6_out); // zero_padding1d_23

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=82
    nnet::conv_1d_cl<layer6_t, layer7_t, config7>(layer6_out, layer7_out, w7, b7); // q_conv1d_66

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=82
    nnet::relu<layer7_t, layer9_t, relu_config9>(layer7_out, layer9_out); // q_activation_66

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=90
    nnet::zeropad1d_cl<layer9_t, layer10_t, config10>(layer9_out, layer10_out); // zero_padding1d_24

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=88
    nnet::conv_1d_cl<layer10_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11); // q_conv1d_67

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=88
    nnet::relu<layer11_t, layer13_t, relu_config13>(layer11_out, layer13_out); // q_activation_67

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=104
    nnet::zeropad1d_cl<layer13_t, layer14_t, config14>(layer13_out, layer14_out); // zero_padding1d_25

    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=102
    nnet::conv_1d_cl<layer14_t, layer15_t, config15>(layer14_out, layer15_out, w15, b15); // q_conv1d_68

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=102
    nnet::relu<layer15_t, layer17_t, relu_config17>(layer15_out, layer17_out); // q_activation_68

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=102
    nnet::pointwise_conv_1d_cl<layer17_t, layer21_t, config21>(layer17_out, layer21_out, w21, b21); // q_conv1d_69

    nnet::hard_sigmoid<layer21_t, result_t, hard_sigmoid_config20>(layer21_out, layer20_out); // q_activation_69

}
