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
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/w7.h"
#include "weights/b7.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/w15.h"
#include "weights/b15.h"
#include "weights/w21.h"
#include "weights/b21.h"

// hls-fpga-machine-learning insert layer-config
// zero_padding1d_22
struct config2 : nnet::padding1d_config {
    static const unsigned in_width = 80;
    static const unsigned n_chan = 1;
    static const unsigned out_width = 82;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv1d_65
struct config3_mult : nnet::dense_config {
    static const unsigned n_in = 3;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 2;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_65_accum_t accum_t;
    typedef bias3_t bias_t;
    typedef weight3_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config3 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 82;
    static const unsigned n_chan = 1;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 80;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 2;
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
    typedef q_conv1d_65_accum_t accum_t;
    typedef bias3_t bias_t;
    typedef weight3_t weight_t;
    typedef config3_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config3::filt_width> config3::pixels[] = {1,3,7,6,4};

// q_activation_65
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = 1280;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_65_table_t table_t;
};

// zero_padding1d_23
struct config6 : nnet::padding1d_config {
    static const unsigned in_width = 80;
    static const unsigned n_chan = 16;
    static const unsigned out_width = 84;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 2;
};

// q_conv1d_66
struct config7_mult : nnet::dense_config {
    static const unsigned n_in = 48;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 38;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_66_accum_t accum_t;
    typedef bias7_t bias_t;
    typedef weight7_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config7 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 84;
    static const unsigned n_chan = 16;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 82;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 38;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 5;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 82;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef q_conv1d_66_accum_t accum_t;
    typedef bias7_t bias_t;
    typedef weight7_t weight_t;
    typedef config7_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config7::filt_width> config7::pixels[] = {1,3,7,6,4};

// q_activation_66
struct relu_config9 : nnet::activ_config {
    static const unsigned n_in = 1312;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_66_table_t table_t;
};

// zero_padding1d_24
struct config10 : nnet::padding1d_config {
    static const unsigned in_width = 82;
    static const unsigned n_chan = 16;
    static const unsigned out_width = 90;
    static const unsigned pad_left = 4;
    static const unsigned pad_right = 4;
};

// q_conv1d_67
struct config11_mult : nnet::dense_config {
    static const unsigned n_in = 48;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 42;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_67_accum_t accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config11 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 90;
    static const unsigned n_chan = 16;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 88;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 42;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 5;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 88;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef q_conv1d_67_accum_t accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    typedef config11_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config11::filt_width> config11::pixels[] = {1,3,7,6,4};

// q_activation_67
struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = 1408;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_67_table_t table_t;
};

// zero_padding1d_25
struct config14 : nnet::padding1d_config {
    static const unsigned in_width = 88;
    static const unsigned n_chan = 16;
    static const unsigned out_width = 104;
    static const unsigned pad_left = 8;
    static const unsigned pad_right = 8;
};

// q_conv1d_68
struct config15_mult : nnet::dense_config {
    static const unsigned n_in = 48;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 42;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_68_accum_t accum_t;
    typedef bias15_t bias_t;
    typedef weight15_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config15 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 104;
    static const unsigned n_chan = 16;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 102;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 42;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 5;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 102;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef q_conv1d_68_accum_t accum_t;
    typedef bias15_t bias_t;
    typedef weight15_t weight_t;
    typedef config15_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config15::filt_width> config15::pixels[] = {1,3,7,6,4};

// q_activation_68
struct relu_config17 : nnet::activ_config {
    static const unsigned n_in = 1632;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef q_activation_68_table_t table_t;
};

// q_conv1d_69
struct config21_mult : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 1;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_69_accum_t accum_t;
    typedef bias21_t bias_t;
    typedef weight21_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config21 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 102;
    static const unsigned n_chan = 16;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 1;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 102;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 1;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 102;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef q_conv1d_69_accum_t accum_t;
    typedef bias21_t bias_t;
    typedef weight21_t weight_t;
    typedef config21_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config21::filt_width> config21::pixels[] = {1};

// q_activation_69
struct hard_sigmoid_config20 {
    static const unsigned n_in = 102;
    static const slope20_t slope;
    static const shift20_t shift;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
};
const slope20_t hard_sigmoid_config20::slope = 0.5;
const shift20_t hard_sigmoid_config20::shift = 0.5;


#endif
