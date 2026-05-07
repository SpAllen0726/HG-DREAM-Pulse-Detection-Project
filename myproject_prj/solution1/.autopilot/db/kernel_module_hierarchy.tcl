set ModuleHierarchy {[{
"Name" : "myproject","ID" : "0","Type" : "dataflow",
"SubInsts" : [
	{"Name" : "zeropad1d_cl_array_ap_fixed_1u_array_ap_fixed_16_6_5_3_0_1u_config18_U0","ID" : "1","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_zeropad1d_cl_array_array_ap_fixed_16_6_5_3_0_1u_config18_Pipeline_CopyMain_fu_36","ID" : "2","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "CopyMain","ID" : "3","Type" : "pipeline"},]},]},
	{"Name" : "conv_1d_cl_array_ap_fixed_1u_array_ap_fixed_27_12_5_3_0_32u_config2_U0","ID" : "4","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "ReadInputWidth","ID" : "5","Type" : "pipeline",
		"SubInsts" : [
		{"Name" : "grp_compute_output_buffer_1d_array_array_ap_fixed_27_12_5_3_0_32u_config2_s_fu_56","ID" : "6","Type" : "pipeline",
				"SubInsts" : [
				{"Name" : "grp_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_27_12_5_3_0_config2_mult_s_fu_68","ID" : "7","Type" : "pipeline"},]},]},]},
	{"Name" : "relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config4_U0","ID" : "8","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "ReLUActLoop","ID" : "9","Type" : "pipeline"},]},
	{"Name" : "zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config19_U0","ID" : "10","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config19_Pipeline_CopyMain_fu_30","ID" : "11","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "CopyMain","ID" : "12","Type" : "pipeline"},]},]},
	{"Name" : "conv_1d_cl_array_array_ap_fixed_22_11_5_3_0_32u_config5_U0","ID" : "13","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "ReadInputWidth","ID" : "14","Type" : "pipeline",
		"SubInsts" : [
		{"Name" : "grp_compute_output_buffer_1d_array_array_ap_fixed_22_11_5_3_0_32u_config5_s_fu_368","ID" : "15","Type" : "pipeline",
				"SubInsts" : [
				{"Name" : "grp_dense_latency_ap_ufixed_6_0_4_0_0_ap_fixed_22_11_5_3_0_config5_mult_s_fu_502","ID" : "16","Type" : "pipeline"},]},]},]},
	{"Name" : "relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config7_U0","ID" : "17","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "ReLUActLoop","ID" : "18","Type" : "pipeline"},]},
	{"Name" : "zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config20_U0","ID" : "19","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_32u_config20_Pipeline_CopyMain_fu_30","ID" : "20","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "CopyMain","ID" : "21","Type" : "pipeline"},]},]},
	{"Name" : "conv_1d_cl_array_array_ap_fixed_22_11_5_3_0_64u_config8_U0","ID" : "22","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "ReadInputWidth","ID" : "23","Type" : "pipeline",
		"SubInsts" : [
		{"Name" : "grp_compute_output_buffer_1d_array_array_ap_fixed_22_11_5_3_0_64u_config8_s_fu_368","ID" : "24","Type" : "pipeline",
				"SubInsts" : [
				{"Name" : "grp_dense_latency_ap_ufixed_6_0_4_0_0_ap_fixed_22_11_5_3_0_config8_mult_s_fu_502","ID" : "25","Type" : "pipeline"},]},]},]},
	{"Name" : "relu_array_ap_fixed_64u_array_ap_ufixed_6_0_4_0_0_64u_relu_config10_U0","ID" : "26","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "ReLUActLoop","ID" : "27","Type" : "pipeline"},]},
	{"Name" : "resize_nearest_array_ap_ufixed_6_0_4_0_0_64u_config11_U0","ID" : "28","Type" : "sequential"},
	{"Name" : "zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_64u_config21_U0","ID" : "29","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_zeropad1d_cl_array_array_ap_ufixed_6_0_4_0_0_64u_config21_Pipeline_CopyMain_fu_30","ID" : "30","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "CopyMain","ID" : "31","Type" : "pipeline"},]},]},
	{"Name" : "conv_1d_cl_array_array_ap_fixed_23_12_5_3_0_32u_config12_U0","ID" : "32","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "ReadInputWidth","ID" : "33","Type" : "pipeline",
		"SubInsts" : [
		{"Name" : "grp_compute_output_buffer_1d_array_array_ap_fixed_23_12_5_3_0_32u_config12_s_fu_688","ID" : "34","Type" : "pipeline",
				"SubInsts" : [
				{"Name" : "grp_dense_latency_ap_ufixed_ap_fixed_23_12_5_3_0_config12_mult_s_fu_950","ID" : "35","Type" : "pipeline"},]},]},]},
	{"Name" : "relu_array_ap_fixed_32u_array_ap_ufixed_6_0_4_0_0_32u_relu_config14_U0","ID" : "36","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "ReLUActLoop","ID" : "37","Type" : "pipeline"},]},
	{"Name" : "pointwise_conv_1d_cl_array_array_ap_fixed_20_9_5_3_0_1u_config22_U0","ID" : "38","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "ReadInputWidth","ID" : "39","Type" : "pipeline"},]},
	{"Name" : "hard_sigmoid_array_array_ap_ufixed_8_0_4_0_0_1u_hard_sigmoid_config17_U0","ID" : "40","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "HardSigmoidActLoop","ID" : "41","Type" : "pipeline"},]},]
}]}