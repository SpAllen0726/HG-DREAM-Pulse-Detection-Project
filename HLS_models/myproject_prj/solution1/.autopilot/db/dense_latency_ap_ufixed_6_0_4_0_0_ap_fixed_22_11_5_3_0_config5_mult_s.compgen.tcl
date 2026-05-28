# This script segment is generated automatically by AutoPilot

set name myproject_mul_6ns_5s_11_1_0
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {mul} IMPL {auto} LATENCY 0 ALLOW_PRAGMA 1
}


set name myproject_mul_6ns_5ns_10_1_0
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {mul} IMPL {auto} LATENCY 0 ALLOW_PRAGMA 1
}


# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 67 \
    name void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_29 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_29 \
    op interface \
    ports { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_29 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 68 \
    name void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_28 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_28 \
    op interface \
    ports { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_28 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 69 \
    name void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_27 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_27 \
    op interface \
    ports { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_27 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 70 \
    name void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_26 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_26 \
    op interface \
    ports { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_26 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 71 \
    name void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_25 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_25 \
    op interface \
    ports { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_25 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 72 \
    name void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_24 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_24 \
    op interface \
    ports { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_24 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 73 \
    name void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_23 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_23 \
    op interface \
    ports { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_23 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 74 \
    name void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_22 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_22 \
    op interface \
    ports { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_22 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 75 \
    name void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_21 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_21 \
    op interface \
    ports { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_21 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 76 \
    name void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_20 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_20 \
    op interface \
    ports { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_20 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 77 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_353 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_353 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_353 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 78 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_354 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_354 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_354 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 79 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_355 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_355 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_355 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 80 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_356 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_356 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_356 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 81 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_357 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_357 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_357 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 82 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_358 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_358 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_358 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 83 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_359 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_359 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_359 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 84 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_360 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_360 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_360 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 85 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_361 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_361 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_361 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 86 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_362 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_362 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_362 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 87 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_363 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_363 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_363 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 88 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_364 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_364 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_364 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 89 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_365 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_365 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_365 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 90 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_366 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_366 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_366 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 91 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_367 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_367 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_367 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 92 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_368 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_368 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_368 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 93 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_369 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_369 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_369 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 94 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_370 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_370 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_370 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 95 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_371 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_371 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_371 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 96 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_372 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_372 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_372 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 97 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_373 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_373 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_373 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 98 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_374 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_374 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_374 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 99 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_375 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_375 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_375 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 100 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_376 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_376 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_376 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 101 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_377 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_377 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_377 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 102 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_378 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_378 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_378 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 103 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_379 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_379 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_379 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 104 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_380 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_380 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_380 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 105 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_381 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_381 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_381 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 106 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_382 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_382 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_382 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 107 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_383 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_383 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_383 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 108 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_384 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_384 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_384 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 109 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_385 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_385 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_385 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 110 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_386 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_386 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_386 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 111 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_387 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_387 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_387 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 112 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_388 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_388 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_388 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 113 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_389 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_389 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_389 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 114 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_390 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_390 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_390 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 115 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_391 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_391 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_391 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 116 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_392 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_392 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_392 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 117 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_393 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_393 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_393 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 118 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_394 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_394 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_394 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 119 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_395 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_395 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_395 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 120 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_396 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_396 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_396 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 121 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_397 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_397 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_397 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 122 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_398 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_398 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_398 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 123 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_399 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_399 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_399 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 124 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_400 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_400 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_400 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 125 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_401 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_401 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_401 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 126 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_402 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_402 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_402 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 127 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_403 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_403 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_403 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 128 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_404 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_404 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_404 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 129 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_405 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_405 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_405 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 130 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_406 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_406 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_406 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 131 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_407 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_407 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_407 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 132 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_408 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_408 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_408 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 133 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_409 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_409 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_409 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 134 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_410 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_410 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_410 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 135 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_411 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_411 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_411 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 136 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_412 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_412 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_412 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 137 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_413 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_413 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_413 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 138 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_414 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_414 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_414 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 139 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_415 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_415 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_415 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 140 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_416 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_416 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_416 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 141 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_417 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_417 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_417 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 142 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_418 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_418 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_418 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 143 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_419 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_419 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_419 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 144 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_420 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_420 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_420 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 145 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_421 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_421 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_421 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 146 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_422 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_422 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_422 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 147 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_423 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_423 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_423 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 148 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_424 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_424 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_424 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 149 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_425 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_425 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_425 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 150 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_426 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_426 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_426 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 151 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_427 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_427 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_427 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 152 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_428 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_428 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_428 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 153 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_429 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_429 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_429 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 154 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_430 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_430 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_430 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 155 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_431 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_431 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_431 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 156 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_432 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_432 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_432 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 157 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_433 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_433 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_433 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 158 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_434 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_434 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_434 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 159 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_435 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_435 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_435 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 160 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_436 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_436 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_436 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 161 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_437 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_437 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_437 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 162 \
    name p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_438 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_438 \
    op interface \
    ports { p_ZZN4nnet24compute_output_buffer_1dINS_5arrayI9ap_ufixedILi6ELi0EL9ap_q_mode4EL9_438 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -1 \
    name ap_return \
    type ap_return \
    reset_level 1 \
    sync_rst true \
    corename ap_return \
    op interface \
    ports { ap_return { O 1 vector } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -2 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_clk \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-113\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}


# Adapter definition:
set PortName ap_rst
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -3 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_rst \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-114\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}


# Adapter definition:
set PortName ap_ce
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_ce] == "cg_default_interface_gen_ce"} {
eval "cg_default_interface_gen_ce { \
    id -4 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_ce \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-113\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}



# merge
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_end
    cg_default_interface_gen_bundle_end
    AESL_LIB_XILADAPTER::native_axis_end
}


