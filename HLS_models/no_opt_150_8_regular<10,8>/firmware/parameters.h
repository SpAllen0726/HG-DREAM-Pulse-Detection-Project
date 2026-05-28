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
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/w14.h"
#include "weights/b14.h"

// hls-fpga-machine-learning insert layer-config
// zp1d_conv1d_17
struct config11 : nnet::padding1d_config {
    static const unsigned in_width = 80;
    static const unsigned n_chan = 1;
    static const unsigned out_width = 82;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv1d_17
struct config2_mult : nnet::dense_config {
    static const unsigned n_in = 3;
    static const unsigned n_out = 8;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef conv1d_17_accum_t accum_t;
    typedef conv1d_17_bias_t bias_t;
    typedef conv1d_17_weight_t weight_t;
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
    static const unsigned n_filt = 8;
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
    typedef conv1d_17_accum_t accum_t;
    typedef conv1d_17_bias_t bias_t;
    typedef conv1d_17_weight_t weight_t;
    typedef config2_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config2::filt_width> config2::pixels[] = {1,3,7,6,4};

// conv1d_17_relu
struct relu_config3 : nnet::activ_config {
    static const unsigned n_in = 640;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef conv1d_17_relu_table_t table_t;
};

// zp1d_conv1d_18
struct config12 : nnet::padding1d_config {
    static const unsigned in_width = 80;
    static const unsigned n_chan = 8;
    static const unsigned out_width = 82;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv1d_18
struct config4_mult : nnet::dense_config {
    static const unsigned n_in = 24;
    static const unsigned n_out = 8;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef conv1d_18_accum_t accum_t;
    typedef conv1d_18_bias_t bias_t;
    typedef conv1d_18_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config4 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 82;
    static const unsigned n_chan = 8;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 8;
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
    typedef conv1d_18_accum_t accum_t;
    typedef conv1d_18_bias_t bias_t;
    typedef conv1d_18_weight_t weight_t;
    typedef config4_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config4::filt_width> config4::pixels[] = {1,3,7,6,4};

// zp1d_conv1d_19
struct config13 : nnet::padding1d_config {
    static const unsigned in_width = 80;
    static const unsigned n_chan = 8;
    static const unsigned out_width = 82;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv1d_19
struct config6_mult : nnet::dense_config {
    static const unsigned n_in = 24;
    static const unsigned n_out = 8;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef conv1d_19_accum_t accum_t;
    typedef conv1d_19_bias_t bias_t;
    typedef conv1d_19_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config6 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 82;
    static const unsigned n_chan = 8;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 8;
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
    typedef conv1d_19_accum_t accum_t;
    typedef conv1d_19_bias_t bias_t;
    typedef conv1d_19_weight_t weight_t;
    typedef config6_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config6::filt_width> config6::pixels[] = {1,3,7,6,4};

// conv1d_20
struct config14_mult : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 1;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef conv1d_20_accum_t accum_t;
    typedef conv1d_20_bias_t bias_t;
    typedef conv1d_20_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config14 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 80;
    static const unsigned n_chan = 8;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 1;
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
    static const unsigned min_width = 1;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 80;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef conv1d_20_accum_t accum_t;
    typedef conv1d_20_bias_t bias_t;
    typedef conv1d_20_weight_t weight_t;
    typedef config14_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config14::filt_width> config14::pixels[] = {1};

// activation_7
struct sigmoid_config10 : nnet::activ_config {
    static const unsigned n_in = 80;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef activation_7_table_t table_t;
};


#endif
