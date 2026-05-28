#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_1,
    hls::stream<result_t> &layer17_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_1,layer17_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 96>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight5_t, 3072>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 32>(b5, "b5.txt");
        nnet::load_weights_from_txt<weight8_t, 6144>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 64>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight12_t, 6144>(w12, "w12.txt");
        nnet::load_weights_from_txt<bias12_t, 32>(b12, "b12.txt");
        nnet::load_weights_from_txt<weight22_t, 32>(w22, "w22.txt");
        nnet::load_weights_from_txt<bias22_t, 1>(b22, "b22.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=82
    nnet::zeropad1d_cl<input_t, layer18_t, config18>(input_1, layer18_out); // zp1d_q_conv1d

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=80
    nnet::conv_1d_cl<layer18_t, layer2_t, config2>(layer18_out, layer2_out, w2, b2); // q_conv1d

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=80
    nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out); // q_activation

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=81
    nnet::zeropad1d_cl<layer4_t, layer19_t, config19>(layer4_out, layer19_out); // zp1d_q_conv1d_1

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=40
    nnet::conv_1d_cl<layer19_t, layer5_t, config5>(layer19_out, layer5_out, w5, b5); // q_conv1d_1

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=40
    nnet::relu<layer5_t, layer7_t, relu_config7>(layer5_out, layer7_out); // q_activation_1

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=42
    nnet::zeropad1d_cl<layer7_t, layer20_t, config20>(layer7_out, layer20_out); // zp1d_q_conv1d_2

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=40
    nnet::conv_1d_cl<layer20_t, layer8_t, config8>(layer20_out, layer8_out, w8, b8); // q_conv1d_2

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=40
    nnet::relu<layer8_t, layer10_t, relu_config10>(layer8_out, layer10_out); // q_activation_2

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=80
    nnet::resize_nearest<layer10_t, config11>(layer10_out, layer11_out); // up_sampling1d

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=82
    nnet::zeropad1d_cl<layer11_t, layer21_t, config21>(layer11_out, layer21_out); // zp1d_q_conv1d_3

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=80
    nnet::conv_1d_cl<layer21_t, layer12_t, config12>(layer21_out, layer12_out, w12, b12); // q_conv1d_3

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=80
    nnet::relu<layer12_t, layer14_t, relu_config14>(layer12_out, layer14_out); // q_activation_3

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=80
    nnet::pointwise_conv_1d_cl<layer14_t, layer22_t, config22>(layer14_out, layer22_out, w22, b22); // q_conv1d_4

    nnet::hard_sigmoid<layer22_t, result_t, hard_sigmoid_config17>(layer22_out, layer17_out); // q_activation_4

}
