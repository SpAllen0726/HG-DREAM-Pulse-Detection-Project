set moduleName dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_27_12_5_3_0_config2_mult_s
set isTopModule 0
set isCombinational 0
set isDatapathOnly 1
set isPipelined 1
set pipeline_type function
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {dense_latency<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<27, 12, 5, 3, 0>, config2_mult>}
set C_modelType { int 864 }
set C_modelArgList {
	{ void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32 int 16 regular {pointer 0} {global 0}  }
	{ void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31 int 16 regular {pointer 0} {global 0}  }
	{ void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30 int 16 regular {pointer 0} {global 0}  }
}
set hasAXIMCache 0
set C_modelArgMapList {[ 
	{ "Name" : "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "ap_return", "interface" : "wire", "bitwidth" : 864} ]}
# RTL Port declarations: 
set portNum 38
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32 sc_in sc_lv 16 signal 0 } 
	{ void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31 sc_in sc_lv 16 signal 1 } 
	{ void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30 sc_in sc_lv 16 signal 2 } 
	{ ap_return_0 sc_out sc_lv 27 signal -1 } 
	{ ap_return_1 sc_out sc_lv 27 signal -1 } 
	{ ap_return_2 sc_out sc_lv 27 signal -1 } 
	{ ap_return_3 sc_out sc_lv 27 signal -1 } 
	{ ap_return_4 sc_out sc_lv 27 signal -1 } 
	{ ap_return_5 sc_out sc_lv 27 signal -1 } 
	{ ap_return_6 sc_out sc_lv 27 signal -1 } 
	{ ap_return_7 sc_out sc_lv 27 signal -1 } 
	{ ap_return_8 sc_out sc_lv 27 signal -1 } 
	{ ap_return_9 sc_out sc_lv 27 signal -1 } 
	{ ap_return_10 sc_out sc_lv 27 signal -1 } 
	{ ap_return_11 sc_out sc_lv 27 signal -1 } 
	{ ap_return_12 sc_out sc_lv 27 signal -1 } 
	{ ap_return_13 sc_out sc_lv 27 signal -1 } 
	{ ap_return_14 sc_out sc_lv 27 signal -1 } 
	{ ap_return_15 sc_out sc_lv 27 signal -1 } 
	{ ap_return_16 sc_out sc_lv 27 signal -1 } 
	{ ap_return_17 sc_out sc_lv 27 signal -1 } 
	{ ap_return_18 sc_out sc_lv 27 signal -1 } 
	{ ap_return_19 sc_out sc_lv 27 signal -1 } 
	{ ap_return_20 sc_out sc_lv 27 signal -1 } 
	{ ap_return_21 sc_out sc_lv 27 signal -1 } 
	{ ap_return_22 sc_out sc_lv 27 signal -1 } 
	{ ap_return_23 sc_out sc_lv 27 signal -1 } 
	{ ap_return_24 sc_out sc_lv 27 signal -1 } 
	{ ap_return_25 sc_out sc_lv 27 signal -1 } 
	{ ap_return_26 sc_out sc_lv 27 signal -1 } 
	{ ap_return_27 sc_out sc_lv 27 signal -1 } 
	{ ap_return_28 sc_out sc_lv 27 signal -1 } 
	{ ap_return_29 sc_out sc_lv 27 signal -1 } 
	{ ap_return_30 sc_out sc_lv 27 signal -1 } 
	{ ap_return_31 sc_out sc_lv 27 signal -1 } 
	{ ap_ce sc_in sc_logic 1 ce -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32", "role": "default" }} , 
 	{ "name": "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31", "role": "default" }} , 
 	{ "name": "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30", "role": "default" }} , 
 	{ "name": "ap_return_0", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_0", "role": "default" }} , 
 	{ "name": "ap_return_1", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_1", "role": "default" }} , 
 	{ "name": "ap_return_2", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_2", "role": "default" }} , 
 	{ "name": "ap_return_3", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_3", "role": "default" }} , 
 	{ "name": "ap_return_4", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_4", "role": "default" }} , 
 	{ "name": "ap_return_5", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_5", "role": "default" }} , 
 	{ "name": "ap_return_6", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_6", "role": "default" }} , 
 	{ "name": "ap_return_7", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_7", "role": "default" }} , 
 	{ "name": "ap_return_8", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_8", "role": "default" }} , 
 	{ "name": "ap_return_9", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_9", "role": "default" }} , 
 	{ "name": "ap_return_10", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_10", "role": "default" }} , 
 	{ "name": "ap_return_11", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_11", "role": "default" }} , 
 	{ "name": "ap_return_12", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_12", "role": "default" }} , 
 	{ "name": "ap_return_13", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_13", "role": "default" }} , 
 	{ "name": "ap_return_14", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_14", "role": "default" }} , 
 	{ "name": "ap_return_15", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_15", "role": "default" }} , 
 	{ "name": "ap_return_16", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_16", "role": "default" }} , 
 	{ "name": "ap_return_17", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_17", "role": "default" }} , 
 	{ "name": "ap_return_18", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_18", "role": "default" }} , 
 	{ "name": "ap_return_19", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_19", "role": "default" }} , 
 	{ "name": "ap_return_20", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_20", "role": "default" }} , 
 	{ "name": "ap_return_21", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_21", "role": "default" }} , 
 	{ "name": "ap_return_22", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_22", "role": "default" }} , 
 	{ "name": "ap_return_23", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_23", "role": "default" }} , 
 	{ "name": "ap_return_24", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_24", "role": "default" }} , 
 	{ "name": "ap_return_25", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_25", "role": "default" }} , 
 	{ "name": "ap_return_26", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_26", "role": "default" }} , 
 	{ "name": "ap_return_27", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_27", "role": "default" }} , 
 	{ "name": "ap_return_28", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_28", "role": "default" }} , 
 	{ "name": "ap_return_29", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_29", "role": "default" }} , 
 	{ "name": "ap_return_30", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_30", "role": "default" }} , 
 	{ "name": "ap_return_31", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_31", "role": "default" }} , 
 	{ "name": "ap_ce", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "ce", "bundle":{"name": "ap_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29"],
		"CDFG" : "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_27_12_5_3_0_config2_mult_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32", "Type" : "None", "Direction" : "I"},
			{"Name" : "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31", "Type" : "None", "Direction" : "I"},
			{"Name" : "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_23_1_0_U6", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_5s_21_1_0_U7", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_22_1_0_U8", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7s_22_1_0_U9", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_22_1_0_U10", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_22_1_0_U11", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_5s_20_1_0_U12", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6s_22_1_0_U13", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6s_22_1_0_U14", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6s_21_1_0_U15", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7s_22_1_0_U16", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6ns_22_1_0_U17", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_5ns_21_1_0_U18", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_22_1_0_U19", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7s_23_1_0_U20", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6ns_22_1_0_U21", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6ns_22_1_0_U22", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_5ns_21_1_0_U23", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_22_1_0_U24", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6ns_22_1_0_U25", "Parent" : "0"},
	{"ID" : "21", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_23_1_0_U26", "Parent" : "0"},
	{"ID" : "22", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_5s_20_1_0_U27", "Parent" : "0"},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6ns_21_1_0_U28", "Parent" : "0"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6ns_21_1_0_U29", "Parent" : "0"},
	{"ID" : "25", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6s_22_1_0_U30", "Parent" : "0"},
	{"ID" : "26", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6s_21_1_0_U31", "Parent" : "0"},
	{"ID" : "27", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_5ns_21_1_0_U32", "Parent" : "0"},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6s_22_1_0_U33", "Parent" : "0"},
	{"ID" : "29", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7s_22_1_0_U34", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_27_12_5_3_0_config2_mult_s {
		void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32 {Type I LastRead 0 FirstWrite -1}
		void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31 {Type I LastRead 0 FirstWrite -1}
		void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30 {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "1", "Max" : "1"}
	, {"Name" : "Interval", "Min" : "1", "Max" : "1"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32 { ap_none {  { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32 in_data 0 16 } } }
	void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31 { ap_none {  { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31 in_data 0 16 } } }
	void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30 { ap_none {  { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30 in_data 0 16 } } }
}
set moduleName dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_27_12_5_3_0_config2_mult_s
set isTopModule 0
set isCombinational 0
set isDatapathOnly 1
set isPipelined 1
set pipeline_type function
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {dense_latency<ap_fixed<16, 6, 5, 3, 0>, ap_fixed<27, 12, 5, 3, 0>, config2_mult>}
set C_modelType { int 864 }
set C_modelArgList {
	{ void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32 int 16 regular {pointer 0} {global 0}  }
	{ void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31 int 16 regular {pointer 0} {global 0}  }
	{ void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30 int 16 regular {pointer 0} {global 0}  }
}
set hasAXIMCache 0
set C_modelArgMapList {[ 
	{ "Name" : "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY", "extern" : 0} , 
 	{ "Name" : "ap_return", "interface" : "wire", "bitwidth" : 864} ]}
# RTL Port declarations: 
set portNum 38
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32 sc_in sc_lv 16 signal 0 } 
	{ void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31 sc_in sc_lv 16 signal 1 } 
	{ void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30 sc_in sc_lv 16 signal 2 } 
	{ ap_return_0 sc_out sc_lv 27 signal -1 } 
	{ ap_return_1 sc_out sc_lv 27 signal -1 } 
	{ ap_return_2 sc_out sc_lv 27 signal -1 } 
	{ ap_return_3 sc_out sc_lv 27 signal -1 } 
	{ ap_return_4 sc_out sc_lv 27 signal -1 } 
	{ ap_return_5 sc_out sc_lv 27 signal -1 } 
	{ ap_return_6 sc_out sc_lv 27 signal -1 } 
	{ ap_return_7 sc_out sc_lv 27 signal -1 } 
	{ ap_return_8 sc_out sc_lv 27 signal -1 } 
	{ ap_return_9 sc_out sc_lv 27 signal -1 } 
	{ ap_return_10 sc_out sc_lv 27 signal -1 } 
	{ ap_return_11 sc_out sc_lv 27 signal -1 } 
	{ ap_return_12 sc_out sc_lv 27 signal -1 } 
	{ ap_return_13 sc_out sc_lv 27 signal -1 } 
	{ ap_return_14 sc_out sc_lv 27 signal -1 } 
	{ ap_return_15 sc_out sc_lv 27 signal -1 } 
	{ ap_return_16 sc_out sc_lv 27 signal -1 } 
	{ ap_return_17 sc_out sc_lv 27 signal -1 } 
	{ ap_return_18 sc_out sc_lv 27 signal -1 } 
	{ ap_return_19 sc_out sc_lv 27 signal -1 } 
	{ ap_return_20 sc_out sc_lv 27 signal -1 } 
	{ ap_return_21 sc_out sc_lv 27 signal -1 } 
	{ ap_return_22 sc_out sc_lv 27 signal -1 } 
	{ ap_return_23 sc_out sc_lv 27 signal -1 } 
	{ ap_return_24 sc_out sc_lv 27 signal -1 } 
	{ ap_return_25 sc_out sc_lv 27 signal -1 } 
	{ ap_return_26 sc_out sc_lv 27 signal -1 } 
	{ ap_return_27 sc_out sc_lv 27 signal -1 } 
	{ ap_return_28 sc_out sc_lv 27 signal -1 } 
	{ ap_return_29 sc_out sc_lv 27 signal -1 } 
	{ ap_return_30 sc_out sc_lv 27 signal -1 } 
	{ ap_return_31 sc_out sc_lv 27 signal -1 } 
	{ ap_ce sc_in sc_logic 1 ce -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32", "role": "default" }} , 
 	{ "name": "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31", "role": "default" }} , 
 	{ "name": "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30", "role": "default" }} , 
 	{ "name": "ap_return_0", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_0", "role": "default" }} , 
 	{ "name": "ap_return_1", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_1", "role": "default" }} , 
 	{ "name": "ap_return_2", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_2", "role": "default" }} , 
 	{ "name": "ap_return_3", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_3", "role": "default" }} , 
 	{ "name": "ap_return_4", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_4", "role": "default" }} , 
 	{ "name": "ap_return_5", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_5", "role": "default" }} , 
 	{ "name": "ap_return_6", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_6", "role": "default" }} , 
 	{ "name": "ap_return_7", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_7", "role": "default" }} , 
 	{ "name": "ap_return_8", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_8", "role": "default" }} , 
 	{ "name": "ap_return_9", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_9", "role": "default" }} , 
 	{ "name": "ap_return_10", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_10", "role": "default" }} , 
 	{ "name": "ap_return_11", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_11", "role": "default" }} , 
 	{ "name": "ap_return_12", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_12", "role": "default" }} , 
 	{ "name": "ap_return_13", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_13", "role": "default" }} , 
 	{ "name": "ap_return_14", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_14", "role": "default" }} , 
 	{ "name": "ap_return_15", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_15", "role": "default" }} , 
 	{ "name": "ap_return_16", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_16", "role": "default" }} , 
 	{ "name": "ap_return_17", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_17", "role": "default" }} , 
 	{ "name": "ap_return_18", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_18", "role": "default" }} , 
 	{ "name": "ap_return_19", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_19", "role": "default" }} , 
 	{ "name": "ap_return_20", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_20", "role": "default" }} , 
 	{ "name": "ap_return_21", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_21", "role": "default" }} , 
 	{ "name": "ap_return_22", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_22", "role": "default" }} , 
 	{ "name": "ap_return_23", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_23", "role": "default" }} , 
 	{ "name": "ap_return_24", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_24", "role": "default" }} , 
 	{ "name": "ap_return_25", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_25", "role": "default" }} , 
 	{ "name": "ap_return_26", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_26", "role": "default" }} , 
 	{ "name": "ap_return_27", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_27", "role": "default" }} , 
 	{ "name": "ap_return_28", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_28", "role": "default" }} , 
 	{ "name": "ap_return_29", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_29", "role": "default" }} , 
 	{ "name": "ap_return_30", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_30", "role": "default" }} , 
 	{ "name": "ap_return_31", "direction": "out", "datatype": "sc_lv", "bitwidth":27, "type": "signal", "bundle":{"name": "ap_return_31", "role": "default" }} , 
 	{ "name": "ap_ce", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "ce", "bundle":{"name": "ap_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29"],
		"CDFG" : "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_27_12_5_3_0_config2_mult_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "1",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32", "Type" : "None", "Direction" : "I"},
			{"Name" : "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31", "Type" : "None", "Direction" : "I"},
			{"Name" : "void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_23_1_0_U6", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_5s_21_1_0_U7", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_22_1_0_U8", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7s_22_1_0_U9", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_22_1_0_U10", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_22_1_0_U11", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_5s_20_1_0_U12", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6s_22_1_0_U13", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6s_22_1_0_U14", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6s_21_1_0_U15", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7s_22_1_0_U16", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6ns_22_1_0_U17", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_5ns_21_1_0_U18", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_22_1_0_U19", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7s_23_1_0_U20", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6ns_22_1_0_U21", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6ns_22_1_0_U22", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_5ns_21_1_0_U23", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_22_1_0_U24", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6ns_22_1_0_U25", "Parent" : "0"},
	{"ID" : "21", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7ns_23_1_0_U26", "Parent" : "0"},
	{"ID" : "22", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_5s_20_1_0_U27", "Parent" : "0"},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6ns_21_1_0_U28", "Parent" : "0"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6ns_21_1_0_U29", "Parent" : "0"},
	{"ID" : "25", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6s_22_1_0_U30", "Parent" : "0"},
	{"ID" : "26", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6s_21_1_0_U31", "Parent" : "0"},
	{"ID" : "27", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_5ns_21_1_0_U32", "Parent" : "0"},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_6s_22_1_0_U33", "Parent" : "0"},
	{"ID" : "29", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_16s_7s_22_1_0_U34", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_27_12_5_3_0_config2_mult_s {
		void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32 {Type I LastRead 0 FirstWrite -1}
		void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31 {Type I LastRead 0 FirstWrite -1}
		void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30 {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "1", "Max" : "1"}
	, {"Name" : "Interval", "Min" : "1", "Max" : "1"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32 { ap_none {  { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_32 in_data 0 16 } } }
	void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31 { ap_none {  { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_31 in_data 0 16 } } }
	void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30 { ap_none {  { void_compute_output_buffer_1d_array_const_stream_weight_t_bias_t_kernel_data_30 in_data 0 16 } } }
}
