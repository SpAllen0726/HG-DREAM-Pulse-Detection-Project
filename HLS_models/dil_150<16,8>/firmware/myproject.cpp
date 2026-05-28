#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_1,
    hls::stream<result_t> &layer16_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_1,layer16_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 96>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight5_t, 3072>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 32>(b5, "b5.txt");
        nnet::load_weights_from_txt<weight8_t, 3072>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 32>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight11_t, 3072>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 32>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight21_t, 32>(w21, "w21.txt");
        nnet::load_weights_from_txt<bias21_t, 1>(b21, "b21.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=82
    nnet::zeropad1d_cl<input_t, layer17_t, config17>(input_1, layer17_out); // zp1d_q_conv1d

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=80
    nnet::conv_1d_cl<layer17_t, layer2_t, config2>(layer17_out, layer2_out, w2, b2); // q_conv1d

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=80
    nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out); // q_activation

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=82
    nnet::zeropad1d_cl<layer4_t, layer18_t, config18>(layer4_out, layer18_out); // zp1d_q_conv1d_1

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=80
    nnet::conv_1d_cl<layer18_t, layer5_t, config5>(layer18_out, layer5_out, w5, b5); // q_conv1d_1

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=80
    nnet::relu<layer5_t, layer7_t, relu_config7>(layer5_out, layer7_out); // q_activation_1

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=82
    nnet::zeropad1d_cl<layer7_t, layer19_t, config19>(layer7_out, layer19_out); // zp1d_q_conv1d_2

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=80
    nnet::conv_1d_cl<layer19_t, layer8_t, config8>(layer19_out, layer8_out, w8, b8); // q_conv1d_2

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=80
    nnet::relu<layer8_t, layer10_t, relu_config10>(layer8_out, layer10_out); // q_activation_2

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=82
    nnet::zeropad1d_cl<layer10_t, layer20_t, config20>(layer10_out, layer20_out); // zp1d_q_conv1d_3

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=80
    nnet::conv_1d_cl<layer20_t, layer11_t, config11>(layer20_out, layer11_out, w11, b11); // q_conv1d_3

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=80
    nnet::relu<layer11_t, layer13_t, relu_config13>(layer11_out, layer13_out); // q_activation_3

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=80
    nnet::pointwise_conv_1d_cl<layer13_t, layer21_t, config21>(layer13_out, layer21_out, w21, b21); // q_conv1d_4

    nnet::hard_sigmoid<layer21_t, result_t, hard_sigmoid_config16>(layer21_out, layer16_out); // q_activation_4

}
