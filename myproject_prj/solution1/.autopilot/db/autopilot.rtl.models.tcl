set SynModuleInfo {
  {SRCNAME zeropad1d_cl<array,array<ap_fixed<16,6,5,3,0>,1u>,config18>_Pipeline_CopyMain MODELNAME zeropad1d_cl_array_array_ap_fixed_16_6_5_3_0_1u_config18_Pipeline_CopyMain RTLNAME myproject_zeropad1d_cl_array_array_ap_fixed_16_6_5_3_0_1u_config18_Pipeline_CopyMain
    SUBMODULES {
      {MODELNAME myproject_flow_control_loop_pipe_sequential_init RTLNAME myproject_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME myproject_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME zeropad1d_cl<array<ap_fixed,1u>,array<ap_fixed<16,6,5,3,0>,1u>,config18> MODELNAME zeropad1d_cl_array_ap_fixed_1u_array_ap_fixed_16_6_5_3_0_1u_config18_s RTLNAME myproject_zeropad1d_cl_array_ap_fixed_1u_array_ap_fixed_16_6_5_3_0_1u_config18_s
    SUBMODULES {
      {MODELNAME myproject_regslice_both RTLNAME myproject_regslice_both BINDTYPE interface TYPE interface_regslice INSTNAME myproject_regslice_both_U}
    }
  }
  {SRCNAME {dense_latency<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<27, 12, 5, 3, 0>, config2_mult>} MODELNAME dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_27_12_5_3_0_config2_mult_s RTLNAME myproject_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_27_12_5_3_0_config2_mult_s
    SUBMODULES {
      {MODELNAME myproject_mul_16s_7ns_23_1_0 RTLNAME myproject_mul_16s_7ns_23_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_5s_21_1_0 RTLNAME myproject_mul_16s_5s_21_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_7ns_22_1_0 RTLNAME myproject_mul_16s_7ns_22_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_7s_22_1_0 RTLNAME myproject_mul_16s_7s_22_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_5s_20_1_0 RTLNAME myproject_mul_16s_5s_20_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_6s_22_1_0 RTLNAME myproject_mul_16s_6s_22_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_6s_21_1_0 RTLNAME myproject_mul_16s_6s_21_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_6ns_22_1_0 RTLNAME myproject_mul_16s_6ns_22_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_5ns_21_1_0 RTLNAME myproject_mul_16s_5ns_21_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_7s_23_1_0 RTLNAME myproject_mul_16s_7s_23_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_16s_6ns_21_1_0 RTLNAME myproject_mul_16s_6ns_21_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME compute_output_buffer_1d<array,array<ap_fixed<27,12,5,3,0>,32u>,config2> MODELNAME compute_output_buffer_1d_array_array_ap_fixed_27_12_5_3_0_32u_config2_s RTLNAME myproject_compute_output_buffer_1d_array_array_ap_fixed_27_12_5_3_0_32u_config2_s}
  {SRCNAME conv_1d_cl<array<ap_fixed,1u>,array<ap_fixed<27,12,5,3,0>,32u>,config2> MODELNAME conv_1d_cl_array_ap_fixed_1u_array_ap_fixed_27_12_5_3_0_32u_config2_s RTLNAME myproject_conv_1d_cl_array_ap_fixed_1u_array_ap_fixed_27_12_5_3_0_32u_config2_s
    SUBMODULES {
      {MODELNAME myproject_flow_control_loop_pipe RTLNAME myproject_flow_control_loop_pipe BINDTYPE interface TYPE internal_upc_flow_control INSTNAME myproject_flow_control_loop_pipe_U}
    }
  }
  {SRCNAME relu<array<ap_fixed,32u>,array<ap_ufixed<6,0,4,0,0>,32u>,relu_config4> MODELNAME relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config4_s RTLNAME myproject_relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config4_s}
  {SRCNAME zeropad1d_cl<array,array<ap_ufixed<6,0,4,0,0>,32u>,config19>_Pipeline_CopyMain MODELNAME zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config19_Pipeline_CopyMain RTLNAME myproject_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config19_Pipeline_CopyMain}
  {SRCNAME zeropad1d_cl<array,array<ap_ufixed<6,0,4,0,0>,32u>,config19> MODELNAME zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config19_s RTLNAME myproject_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config19_s}
  {SRCNAME {dense_latency<ap_ufixed<6, 0, 4, 0, 0>, ap_fixed<22, 11, 5, 3, 0>, config5_mult>} MODELNAME dense_latency_ap_ufixed_6_0_4_0_0_ap_fixed_22_11_5_3_0_config5_mult_s RTLNAME myproject_dense_latency_ap_ufixed_6_0_4_0_0_ap_fixed_22_11_5_3_0_config5_mult_s
    SUBMODULES {
      {MODELNAME myproject_mul_6ns_5s_11_1_0 RTLNAME myproject_mul_6ns_5s_11_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME myproject_mul_6ns_5ns_10_1_0 RTLNAME myproject_mul_6ns_5ns_10_1_0 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME compute_output_buffer_1d<array,array<ap_fixed<22,11,5,3,0>,32u>,config5> MODELNAME compute_output_buffer_1d_array_array_ap_fixed_22_11_5_3_0_32u_config5_s RTLNAME myproject_compute_output_buffer_1d_array_array_ap_fixed_22_11_5_3_0_32u_config5_s}
  {SRCNAME conv_1d_cl<array,array<ap_fixed<22,11,5,3,0>,32u>,config5> MODELNAME conv_1d_cl_array_array_ap_fixed_22_11_5_3_0_32u_config5_s RTLNAME myproject_conv_1d_cl_array_array_ap_fixed_22_11_5_3_0_32u_config5_s}
  {SRCNAME relu<array<ap_fixed,32u>,array<ap_ufixed<6,0,4,0,0>,32u>,relu_config7> MODELNAME relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config7_s RTLNAME myproject_relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config7_s}
  {SRCNAME zeropad1d_cl<array,array<ap_ufixed<6,0,4,0,0>,32u>,config20>_Pipeline_CopyMain MODELNAME zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config20_Pipeline_CopyMain RTLNAME myproject_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config20_Pipeline_CopyMain}
  {SRCNAME zeropad1d_cl<array,array<ap_ufixed<6,0,4,0,0>,32u>,config20> MODELNAME zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config20_s RTLNAME myproject_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config20_s}
  {SRCNAME {dense_latency<ap_ufixed<6, 0, 4, 0, 0>, ap_fixed<22, 11, 5, 3, 0>, config8_mult>} MODELNAME dense_latency_ap_ufixed_6_0_4_0_0_ap_fixed_22_11_5_3_0_config8_mult_s RTLNAME myproject_dense_latency_ap_ufixed_6_0_4_0_0_ap_fixed_22_11_5_3_0_config8_mult_s}
  {SRCNAME compute_output_buffer_1d<array,array<ap_fixed<22,11,5,3,0>,64u>,config8> MODELNAME compute_output_buffer_1d_array_array_ap_fixed_22_11_5_3_0_64u_config8_s RTLNAME myproject_compute_output_buffer_1d_array_array_ap_fixed_22_11_5_3_0_64u_config8_s}
  {SRCNAME conv_1d_cl<array,array<ap_fixed<22,11,5,3,0>,64u>,config8> MODELNAME conv_1d_cl_array_array_ap_fixed_22_11_5_3_0_64u_config8_s RTLNAME myproject_conv_1d_cl_array_array_ap_fixed_22_11_5_3_0_64u_config8_s}
  {SRCNAME relu<array<ap_fixed,64u>,array<ap_ufixed<6,0,4,0,0>,64u>,relu_config10> MODELNAME relu_array_ap_fixed_64u_array_ap_ufixed_6_0_4_0_0_64u_relu_config10_s RTLNAME myproject_relu_array_ap_fixed_64u_array_ap_ufixed_6_0_4_0_0_64u_relu_config10_s}
  {SRCNAME {resize_nearest<array<ap_ufixed<6, 0, 4, 0, 0>, 64u>, config11>} MODELNAME resize_nearest_array_ap_ufixed_6_0_4_0_0_64u_config11_s RTLNAME myproject_resize_nearest_array_ap_ufixed_6_0_4_0_0_64u_config11_s}
  {SRCNAME zeropad1d_cl<array,array<ap_ufixed<6,0,4,0,0>,64u>,config21>_Pipeline_CopyMain MODELNAME zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_64u_config21_Pipeline_CopyMain RTLNAME myproject_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_64u_config21_Pipeline_CopyMain}
  {SRCNAME zeropad1d_cl<array,array<ap_ufixed<6,0,4,0,0>,64u>,config21> MODELNAME zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_64u_config21_s RTLNAME myproject_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_64u_config21_s}
  {SRCNAME dense_latency<ap_ufixed,ap_fixed<23,12,5,3,0>,config12_mult> MODELNAME dense_latency_ap_ufixed_ap_fixed_23_12_5_3_0_config12_mult_s RTLNAME myproject_dense_latency_ap_ufixed_ap_fixed_23_12_5_3_0_config12_mult_s}
  {SRCNAME compute_output_buffer_1d<array,array<ap_fixed<23,12,5,3,0>,32u>,config12> MODELNAME compute_output_buffer_1d_array_array_ap_fixed_23_12_5_3_0_32u_config12_s RTLNAME myproject_compute_output_buffer_1d_array_array_ap_fixed_23_12_5_3_0_32u_config12_s}
  {SRCNAME conv_1d_cl<array,array<ap_fixed<23,12,5,3,0>,32u>,config12> MODELNAME conv_1d_cl_array_array_ap_fixed_23_12_5_3_0_32u_config12_s RTLNAME myproject_conv_1d_cl_array_array_ap_fixed_23_12_5_3_0_32u_config12_s}
  {SRCNAME relu<array<ap_fixed,32u>,array<ap_ufixed<6,0,4,0,0>,32u>,relu_config14> MODELNAME relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config14_s RTLNAME myproject_relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config14_s}
  {SRCNAME pointwise_conv_1d_cl<array,array<ap_fixed<20,9,5,3,0>,1u>,config22> MODELNAME pointwise_conv_1d_cl_array_array_ap_fixed_20_9_5_3_0_1u_config22_s RTLNAME myproject_pointwise_conv_1d_cl_array_array_ap_fixed_20_9_5_3_0_1u_config22_s}
  {SRCNAME hard_sigmoid<array,array<ap_ufixed<8,0,4,0,0>,1u>,hard_sigmoid_config17> MODELNAME hard_sigmoid_array_array_ap_ufixed_8_0_4_0_0_1u_hard_sigmoid_config17_s RTLNAME myproject_hard_sigmoid_array_array_ap_ufixed_8_0_4_0_0_1u_hard_sigmoid_config17_s}
  {SRCNAME myproject MODELNAME myproject RTLNAME myproject IS_TOP 1
    SUBMODULES {
      {MODELNAME myproject_fifo_w16_d82_A RTLNAME myproject_fifo_w16_d82_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer18_out_U}
      {MODELNAME myproject_fifo_w864_d80_A RTLNAME myproject_fifo_w864_d80_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer2_out_U}
      {MODELNAME myproject_fifo_w192_d80_A RTLNAME myproject_fifo_w192_d80_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer4_out_U}
      {MODELNAME myproject_fifo_w192_d81_A RTLNAME myproject_fifo_w192_d81_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer19_out_U}
      {MODELNAME myproject_fifo_w704_d40_A RTLNAME myproject_fifo_w704_d40_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer5_out_U}
      {MODELNAME myproject_fifo_w192_d40_A RTLNAME myproject_fifo_w192_d40_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer7_out_U}
      {MODELNAME myproject_fifo_w192_d42_A RTLNAME myproject_fifo_w192_d42_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer20_out_U}
      {MODELNAME myproject_fifo_w1408_d40_A RTLNAME myproject_fifo_w1408_d40_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer8_out_U}
      {MODELNAME myproject_fifo_w384_d40_A RTLNAME myproject_fifo_w384_d40_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer10_out_U}
      {MODELNAME myproject_fifo_w384_d80_A RTLNAME myproject_fifo_w384_d80_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer11_out_U}
      {MODELNAME myproject_fifo_w384_d82_A RTLNAME myproject_fifo_w384_d82_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer21_out_U}
      {MODELNAME myproject_fifo_w736_d80_A RTLNAME myproject_fifo_w736_d80_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer12_out_U}
      {MODELNAME myproject_fifo_w192_d80_A RTLNAME myproject_fifo_w192_d80_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer14_out_U}
      {MODELNAME myproject_fifo_w20_d80_A RTLNAME myproject_fifo_w20_d80_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME layer15_out_U}
      {MODELNAME myproject_start_for_conv_1d_cl_array_ap_fixed_1u_array_ap_fixed_27_12_5_3_0_32u_config2bkb RTLNAME myproject_start_for_conv_1d_cl_array_ap_fixed_1u_array_ap_fixed_27_12_5_3_0_32u_config2bkb BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_conv_1d_cl_array_ap_fixed_1u_array_ap_fixed_27_12_5_3_0_32u_config2bkb_U}
      {MODELNAME myproject_start_for_relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config4_U0 RTLNAME myproject_start_for_relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config4_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config4_U0_U}
      {MODELNAME myproject_start_for_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config19_U0 RTLNAME myproject_start_for_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config19_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config19_U0_U}
      {MODELNAME myproject_start_for_conv_1d_cl_array_array_ap_fixed_22_11_5_3_0_32u_config5_U0 RTLNAME myproject_start_for_conv_1d_cl_array_array_ap_fixed_22_11_5_3_0_32u_config5_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_conv_1d_cl_array_array_ap_fixed_22_11_5_3_0_32u_config5_U0_U}
      {MODELNAME myproject_start_for_relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config7_U0 RTLNAME myproject_start_for_relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config7_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config7_U0_U}
      {MODELNAME myproject_start_for_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config20_U0 RTLNAME myproject_start_for_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config20_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config20_U0_U}
      {MODELNAME myproject_start_for_conv_1d_cl_array_array_ap_fixed_22_11_5_3_0_64u_config8_U0 RTLNAME myproject_start_for_conv_1d_cl_array_array_ap_fixed_22_11_5_3_0_64u_config8_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_conv_1d_cl_array_array_ap_fixed_22_11_5_3_0_64u_config8_U0_U}
      {MODELNAME myproject_start_for_relu_array_ap_fixed_64u_array_ap_ufixed_6_0_4_0_0_64u_relu_config10cud RTLNAME myproject_start_for_relu_array_ap_fixed_64u_array_ap_ufixed_6_0_4_0_0_64u_relu_config10cud BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_relu_array_ap_fixed_64u_array_ap_ufixed_6_0_4_0_0_64u_relu_config10cud_U}
      {MODELNAME myproject_start_for_resize_nearest_array_ap_ufixed_6_0_4_0_0_64u_config11_U0 RTLNAME myproject_start_for_resize_nearest_array_ap_ufixed_6_0_4_0_0_64u_config11_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_resize_nearest_array_ap_ufixed_6_0_4_0_0_64u_config11_U0_U}
      {MODELNAME myproject_start_for_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_64u_config21_U0 RTLNAME myproject_start_for_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_64u_config21_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_64u_config21_U0_U}
      {MODELNAME myproject_start_for_conv_1d_cl_array_array_ap_fixed_23_12_5_3_0_32u_config12_U0 RTLNAME myproject_start_for_conv_1d_cl_array_array_ap_fixed_23_12_5_3_0_32u_config12_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_conv_1d_cl_array_array_ap_fixed_23_12_5_3_0_32u_config12_U0_U}
      {MODELNAME myproject_start_for_relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config14dEe RTLNAME myproject_start_for_relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config14dEe BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config14dEe_U}
      {MODELNAME myproject_start_for_pointwise_conv_1d_cl_array_array_ap_fixed_20_9_5_3_0_1u_config22_U0 RTLNAME myproject_start_for_pointwise_conv_1d_cl_array_array_ap_fixed_20_9_5_3_0_1u_config22_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_pointwise_conv_1d_cl_array_array_ap_fixed_20_9_5_3_0_1u_config22_U0_U}
      {MODELNAME myproject_start_for_hard_sigmoid_array_array_ap_ufixed_8_0_4_0_0_1u_hard_sigmoid_configeOg RTLNAME myproject_start_for_hard_sigmoid_array_array_ap_ufixed_8_0_4_0_0_1u_hard_sigmoid_configeOg BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_hard_sigmoid_array_array_ap_ufixed_8_0_4_0_0_1u_hard_sigmoid_configeOg_U}
    }
  }
}
