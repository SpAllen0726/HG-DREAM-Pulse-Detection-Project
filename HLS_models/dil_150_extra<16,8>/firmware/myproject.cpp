#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_4,
    hls::stream<result_t> &layer19_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_4,layer19_out 
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
        nnet::load_weights_from_txt<weight14_t, 3072>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 32>(b14, "b14.txt");
        nnet::load_weights_from_txt<weight25_t, 32>(w25, "w25.txt");
        nnet::load_weights_from_txt<bias25_t, 1>(b25, "b25.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=82
    nnet::zeropad1d_cl<input_t, layer20_t, config20>(input_4, layer20_out); // zp1d_q_conv1d_17

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=80
    nnet::conv_1d_cl<layer20_t, layer2_t, config2>(layer20_out, layer2_out, w2, b2); // q_conv1d_17

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=80
    nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out); // q_activation_17

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=82
    nnet::zeropad1d_cl<layer4_t, layer21_t, config21>(layer4_out, layer21_out); // zp1d_q_conv1d_18

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=80
    nnet::conv_1d_cl<layer21_t, layer5_t, config5>(layer21_out, layer5_out, w5, b5); // q_conv1d_18

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=80
    nnet::relu<layer5_t, layer7_t, relu_config7>(layer5_out, layer7_out); // q_activation_18

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=82
    nnet::zeropad1d_cl<layer7_t, layer22_t, config22>(layer7_out, layer22_out); // zp1d_q_conv1d_19

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=80
    nnet::conv_1d_cl<layer22_t, layer8_t, config8>(layer22_out, layer8_out, w8, b8); // q_conv1d_19

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=80
    nnet::relu<layer8_t, layer10_t, relu_config10>(layer8_out, layer10_out); // q_activation_19

    hls::stream<layer23_t> layer23_out("layer23_out");
    #pragma HLS STREAM variable=layer23_out depth=82
    nnet::zeropad1d_cl<layer10_t, layer23_t, config23>(layer10_out, layer23_out); // zp1d_q_conv1d_20

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=80
    nnet::conv_1d_cl<layer23_t, layer11_t, config11>(layer23_out, layer11_out, w11, b11); // q_conv1d_20

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=80
    nnet::relu<layer11_t, layer13_t, relu_config13>(layer11_out, layer13_out); // q_activation_20

    hls::stream<layer24_t> layer24_out("layer24_out");
    #pragma HLS STREAM variable=layer24_out depth=82
    nnet::zeropad1d_cl<layer13_t, layer24_t, config24>(layer13_out, layer24_out); // zp1d_q_conv1d_21

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=80
    nnet::conv_1d_cl<layer24_t, layer14_t, config14>(layer24_out, layer14_out, w14, b14); // q_conv1d_21

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=80
    nnet::relu<layer14_t, layer16_t, relu_config16>(layer14_out, layer16_out); // q_activation_21

    hls::stream<layer25_t> layer25_out("layer25_out");
    #pragma HLS STREAM variable=layer25_out depth=80
    nnet::pointwise_conv_1d_cl<layer16_t, layer25_t, config25>(layer16_out, layer25_out, w25, b25); // q_conv1d_22

    nnet::hard_sigmoid<layer25_t, result_t, hard_sigmoid_config19>(layer25_out, layer19_out); // q_activation_22

}
