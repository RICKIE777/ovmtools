# OpenVINO model tools


## Performance regression test for CPU plugin

## mount IR cache by nfs

```bash

sudo apt-get install nfs-common

mkdir nfs
sudo mount -t nfs 10.67.108.173:/home/vsi/nfs_share ./nfs

```

## source code prepare

### 1. Modify CMakeLists.txt
```
src/plugins/intel_cpu/thirdparty/CMakeLists.txt
Switching DNNL_BUILD_TESTS OFF to ON
set(DNNL_BUILD_TESTS ON CACHE BOOL "" FORCE) 
```

### 2. If you want to use "USE_BRG" environments
Add the following code fo file *src/plugins/intel_cpu/src/nodes/conv.cpp*
```
Add code at the end of Convolution::initTryBrgconvFlag() function
auto p = getenv("USE_BRG");
if (p) {
    shouldTryBrgconv = p[0] == '1';
} 
```

## configuration profile -- config.ini
***Tips: If you do not use some non-bool parameters, please set them to empty, do not delete keywords. Such as binB="". If you 
want to set bool parameters, please use `True` or `False`. Pay attention to capitalization.***

### 1. Common configuration
- Compare two versions of binary. 

The output is 
1. Compare the model data with fps lower than 5% under the two versions of binary, binA_prefix.vs.binB_prefix.max.csv,
"model,binA_prefix.f32,binB_prefix.f32,ratio (A-B)/B,geomean" d
2. For models with fps below 5%, show the layer with the most difference in each model, 
binA_prefix.layer.csv, "name,binA_prefix.f32_fps,binB_prefix.f32_fps,ratio,delta(ms),layer1,time1(ms),,,,,,,,,,"
3. For models with fps below 5%, show the running time of each layer of each model,detail_layer_binA_prefix_binB_prefix.csv, 
"model_name,layer_name,layer_type,binA_prefix,binA_prefix_benchdnn_cmd,binA_prefix_time,binA_prefix_benchdnn_time_average,binA_prefix_benchdnn_time_min,binB_prefix,binB_prefix_cmd,binB_prefix_time,binB_prefix_benchdnn_time_average,binB_prefix_benchdnn_time_min,(binA_prefix_time - binB_prefix_time)/binB_prefix_time,(binA_prefix_benchdnn_time_average - binB_prefix_benchdnn_time_average)/binB_prefix_benchdnn_time_average,(binA_prefix_benchdnn_time_min - binB_prefix_benchdnn_time_min)/binB_prefix_benchdnn_time_min
" 
```
[Basic]
cpus = 4,5
node = 0
run_times = 2

[BIN]
binA = /home/ruiqiyang/disk/openVINO/test_cc_openvino/openvino/bin/intel64/Debug/
binB = /home/ruiqiyang/disk/openVINO/openvino/bin/intel64/Release/

[FP32]
model_path=/home/ruiqiyang/disk/openVINO/ovmtools/fp32_test_model
binA_prefix=brg.f32
binB_prefix=jit.f32
common_args= -infer_precision=f32 -hint none -t=5 -b=1 -nireq=1 -nstreams=1 -nthreads=4 -json_stats -report_type=detailed_counters

[INT8]
model_path=/home/ruiqiyang/disk/openVINO/ovmtools/int8_test_model
binA_prefix=brg.i8
binB_prefix=jit.i8
common_args=-hint none -t=5 -b=1 -nireq=1 -nstreams=1 -nthreads=4 -json_stats -report_type=detailed_counters

[Mode]
single=False
numactl=True
need_run_benchdnn=False
VERBOSE_CONVERT=/src/plugins/intel_cpu/thirdparty/onednn/scripts/verbose_converter
diff_bin=True
diff_env=False
envA=""
envB=""

[Filter]
need_filter=True
threshold=-0.05

```
- Compare two environments of binary
```
[Basic]
cpus = 4,5
node = 0
run_times = 2

[BIN]
binA = /home/ruiqiyang/disk/openVINO/test_cc_openvino/openvino/bin/intel64/Debug/
binB = 

[FP32]
model_path=/home/ruiqiyang/disk/openVINO/ovmtools/fp32_test_model
binA_prefix=brg.f32
binB_prefix=jit.f32
common_args= -infer_precision=f32 -hint none -t=5 -b=1 -nireq=1 -nstreams=1 -nthreads=4 -json_stats -report_type=detailed_counters

[INT8]
model_path=/home/ruiqiyang/disk/openVINO/ovmtools/int8_test_model
binA_prefix=brg.i8
binB_prefix=jit.i8
common_args=-hint none -t=5 -b=1 -nireq=1 -nstreams=1 -nthreads=4 -json_stats -report_type=detailed_counters

[Mode]
single=False
numactl=True
need_run_benchdnn=False
VERBOSE_CONVERT=/src/plugins/intel_cpu/thirdparty/onednn/scripts/verbose_converter
diff_bin=False

diff_env=True
envA=USE_BRG=1
envB=USE_BRG=0

[Filter]
need_filter=True
threshold=-0.05

```
- No comparison
```
[Basic]
cpus = 4,5
node = 0
run_times = 2

[BIN]
binA = /home/ruiqiyang/disk/openVINO/test_cc_openvino/openvino/bin/intel64/Debug/
binB = /home/ruiqiyang/disk/openVINO/openvino/bin/intel64/Release/

[FP32]
model_path=/home/ruiqiyang/disk/openVINO/ovmtools/fp32_test_model
binA_prefix=brg.f32
binB_prefix=jit.f32
common_args= -infer_precision=f32 -hint none -t=5 -b=1 -nireq=1 -nstreams=1 -nthreads=4 -json_stats -report_type=detailed_counters

[INT8]
model_path=/home/ruiqiyang/disk/openVINO/ovmtools/int8_test_model
binA_prefix=brg.i8
binB_prefix=jit.i8
common_args=-hint none -t=5 -b=1 -nireq=1 -nstreams=1 -nthreads=4 -json_stats -report_type=detailed_counters

[Mode]
single=True
numactl=True

need_run_benchdnn=False
VERBOSE_CONVERT=/src/plugins/intel_cpu/thirdparty/onednn/scripts/verbose_converter
diff_bin=
diff_env=
envA=USE_BRG=1
envB=USE_BRG=0

[Filter]
# Filtering models with reduced FPS
need_filter=True
threshold=-0.05

```

### 2. Support single model in the same environment. No comparison

If you only want to display the situation under the fixed environment variables of a model, please configure as follows
```
[Mode]
# There is no comparison, only use one binary(binA).
single=True
```

### 3. Support for comparing the performance results of two versions of binary
```
[Mode]
single=False
diff_bin=True

# If you want to filter data
[Filter]
need_filter=True
threshold=-0.05

# If you don't want to filter model, you want to show all the model data
[Filter]
need_filter=True
threshold=0
```
### 4. Support for comparing the performance results of two different environments
```
[Mode]
single=False
# Compare different env, such as BRG
diff_env=True

# split by =
envA=USE_BRG=1
envB=USE_BRG=0
```
 - using the same binary
```
[BIN]
binA = /home/openvino/bin/intel64/Debug/
binB = ""
```
 - using different binaries
```
# binA corresponds to envA
# binB corresponds to envB
[BIN]
binA = /home/1/openvino/bin/intel64/Debug/ 
binB = /home/2/openvino/bin/intel64/Debug/
 ```
### 5. Some parameters
`run_times`:  The number of times to run benchmark_app. We will take the one with the largest fps to create the output

`envA=USE_BRG=1,DNNL-ARCH=AVX2`: The environment GroupA. Each environment variable is separated by ","

`envB`: The environment GroupB.

## run tests
After setting the config.ini
```
python3 main.py
```

