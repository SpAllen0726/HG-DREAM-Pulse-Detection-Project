#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_conv1d_stream.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"
#include "nnet_utils/nnet_sepconv1d_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w5.h"
#include "weights/b5.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/w14.h"
#include "weights/b14.h"
#include "weights/w25.h"
#include "weights/b25.h"

// hls-fpga-machine-learning insert layer-config
// zp1d_q_conv1d_17
struct config20 : nnet::padding1d_config {
    static const unsigned in_width = 80;
    static const unsigned n_chan = 1;
    static const unsigned out_width = 82;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv1d_17
struct config2_mult : nnet::dense_config {
    static const unsigned n_in = 3;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_17_accum_t accum_t;
    typedef bias2_t bias_t;
    typedef weight2_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 82;
    static const unsigned n_chan = 1;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 80;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 5;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 80;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef q_conv1d_17_accum_t accum_t;
    typedef bias2_t bias_t;
    typedef weight2_t weight_t;
    typedef config2_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config2::filt_width> config2::pixels[] = {1,3,7,6,4};

// q_activation_17
struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = 2560;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_17_table_t table_t;
};

// zp1d_q_conv1d_18
struct config21 : nnet::padding1d_config {
    static const unsigned in_width = 80;
    static const unsigned n_chan = 32;
    static const unsigned out_width = 82;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv1d_18
struct config5_mult : nnet::dense_config {
    static const unsigned n_in = 96;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 266;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_18_accum_t accum_t;
    typedef bias5_t bias_t;
    typedef weight5_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config5 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 82;
    static const unsigned n_chan = 32;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 80;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 266;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 5;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 80;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef q_conv1d_18_accum_t accum_t;
    typedef bias5_t bias_t;
    typedef weight5_t weight_t;
    typedef config5_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config5::filt_width> config5::pixels[] = {1,3,7,6,4};

// q_activation_18
struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = 2560;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_18_table_t table_t;
};

// zp1d_q_conv1d_19
struct config22 : nnet::padding1d_config {
    static const unsigned in_width = 80;
    static const unsigned n_chan = 32;
    static const unsigned out_width = 82;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv1d_19
struct config8_mult : nnet::dense_config {
    static const unsigned n_in = 96;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 234;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_19_accum_t accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config8 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 82;
    static const unsigned n_chan = 32;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 80;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 234;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 5;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 80;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef q_conv1d_19_accum_t accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef config8_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config8::filt_width> config8::pixels[] = {1,3,7,6,4};

// q_activation_19
struct relu_config10 : nnet::activ_config {
    static const unsigned n_in = 2560;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_19_table_t table_t;
};

// zp1d_q_conv1d_20
struct config23 : nnet::padding1d_config {
    static const unsigned in_width = 80;
    static const unsigned n_chan = 32;
    static const unsigned out_width = 82;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv1d_20
struct config11_mult : nnet::dense_config {
    static const unsigned n_in = 96;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 249;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_20_accum_t accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config11 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 82;
    static const unsigned n_chan = 32;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 80;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 249;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 5;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 80;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef q_conv1d_20_accum_t accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    typedef config11_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config11::filt_width> config11::pixels[] = {1,3,7,6,4};

// q_activation_20
struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = 2560;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_20_table_t table_t;
};

// zp1d_q_conv1d_21
struct config24 : nnet::padding1d_config {
    static const unsigned in_width = 80;
    static const unsigned n_chan = 32;
    static const unsigned out_width = 82;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv1d_21
struct config14_mult : nnet::dense_config {
    static const unsigned n_in = 96;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 239;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_21_accum_t accum_t;
    typedef bias14_t bias_t;
    typedef weight14_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config14 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 82;
    static const unsigned n_chan = 32;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 80;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 239;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 5;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 80;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef q_conv1d_21_accum_t accum_t;
    typedef bias14_t bias_t;
    typedef weight14_t weight_t;
    typedef config14_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config14::filt_width> config14::pixels[] = {1,3,7,6,4};

// q_activation_21
struct relu_config16 : nnet::activ_config {
    static const unsigned n_in = 2560;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_21_table_t table_t;
};

// q_conv1d_22
struct config25_mult : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 1;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_22_accum_t accum_t;
    typedef bias25_t bias_t;
    typedef weight25_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config25 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 80;
    static const unsigned n_chan = 32;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 1;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 80;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 1;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 1;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 80;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef q_conv1d_22_accum_t accum_t;
    typedef bias25_t bias_t;
    typedef weight25_t weight_t;
    typedef config25_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config25::filt_width> config25::pixels[] = {1};

// q_activation_22
struct hard_sigmoid_config19 {
    static const unsigned n_in = 80;
    static const slope19_t slope;
    static const shift19_t shift;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
};
const slope19_t hard_sigmoid_config19::slope = 0.5;
const shift19_t hard_sigmoid_config19::shift = 0.5;


#endif
