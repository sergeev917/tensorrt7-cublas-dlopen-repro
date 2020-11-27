Steps to reproduce
==================

1. Download and place `cuda-repo-ubuntu1604-11-1-local_11.1.1-455.32.00-1_amd64.deb`
into the `vendor` subdirectory.

2. Download and place `cudnn-11.1-linux-x64-v8.0.5.39.tgz` into the `vendor` subdirectory.

3. Download and place `TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz`
into the `vendor` subdirectory.

4. Run `./build_and_run.sh`. The script will build a docker image and run it.
The expected result of the running part looks like:

```
[...]
Eliminating concatenation ssd0_concat2
Generating copy for ssd0_reshape1 to ssd0_concat2
Generating copy for ssd0_reshape0 to ssd0_concat2
After concat removal: 69 layers
Graph construction and optimization completed in 0.0473554 seconds.
Unable to load library: libcublasLt.so.11
Assertion failed: mHandle
/_src/common/libUtil.h:44
Aborting...
/_src/common/libUtil.h (44) - Assertion Error in DynamicLibrary: 0 (mHandle)
IBuilder::buildCudaEngine() failed
```

Details
=======

The program run have tried to `dlopen()` `libcublasLt.so.11` shared library
despite cublas being linked-in as a static library.

Note that `libmyelin_executor_static.a` static library (that is a part
of TensorRT distribution) imports symbols from both cublas and cublas:

```
libmyelin_executor_static.a(exec_instruction.o): In function `_ZNK6myelin4exec10ixn_gemm_t4execEv':
exec_instruction.cpp:(.text+0x6a2): undefined reference to `cublasSetStream_v2'
exec_instruction.cpp:(.text+0x7ea): undefined reference to `cublasLtMatmul'
exec_instruction.cpp:(.text+0x11ed): undefined reference to `cublasGemmStridedBatchedEx'
exec_instruction.cpp:(.text+0x13ce): undefined reference to `cublasHgemmStridedBatched'
exec_instruction.cpp:(.text+0x147f): undefined reference to `cublasGetMathMode'
exec_instruction.cpp:(.text+0x15fb): undefined reference to `cublasSgemmStridedBatched'
exec_instruction.cpp:(.text+0x16ed): undefined reference to `cublasGemmEx'
exec_instruction.cpp:(.text+0x17f6): undefined reference to `cublasGetMathMode'
exec_instruction.cpp:(.text+0x1ac8): undefined reference to `cublasSgemmEx'
```

Errors above can be seen when `libmyelin_executor_static.a` is linked to
an executable without supplying `libcublas_static.a` and `libcublasLt_static.a`.

This means that cublas and cublaslt symbols were imported by the static tensorrt
libraries, but then later at runtime there are still attempts to load `libcublasLt.so.11`
from the tensorrt implementation.

The precise point of `dlopen` call is illustrated here:
```
#0  __dlopen (file=0x5555cc7d5558 "libcublasLt.so.11", mode=1) at dlopen.c:75
#1  0x000055557a8969f2 in nvinfer1::CublasLtWrapper::CublasLtWrapper() ()
#2  0x000055557a88a50e in nvinfer1::rt::initializeCommonContext(nvinfer1::rt::CommonContext&, nvinfer1::IGpuAllocator&, unsigned int) ()
#3  0x000055557a93e3c0 in nvinfer1::builder::(anonymous namespace)::makeEngineFromGraph(nvinfer1::Network const&, nvinfer1::NetworkBuildConfig const&, nvinfer1::NetworkQuantizationConfig const&, nvinfer1::builder::EngineBuildContext const&, nvinfer1::builder::Graph&, std::map<int, std::unordered_map<std::string, std::vector<nvinfer1::builder::DynamicRangeSymbol, std::allocator<nvinfer1::builder::DynamicRangeSymbol> >, std::hash<std::string>, std::equal_to<std::string>, std::allocator<std::pair<std::string const, std::vector<nvinfer1::builder::DynamicRangeSymbol, std::allocator<nvinfer1::builder::DynamicRangeSymbol> > > > >, std::less<int>, std::allocator<std::pair<int const, std::unordered_map<std::string, std::vector<nvinfer1::builder::DynamicRangeSymbol, std::allocator<nvinfer1::builder::DynamicRangeSymbol> >, std::hash<std::string>, std::equal_to<std::string>, std::allocator<std::pair<std::string const, std::vector<nvinfer1::builder::DynamicRangeSymbol, std::allocator<nvinfer1::builder::DynamicRangeSymbol> > > > > > > >*, int, bool, bool) [clone .constprop.1430] ()
#4  0x000055557a944316 in nvinfer1::builder::buildEngine(nvinfer1::NetworkBuildConfig&, nvinfer1::NetworkQuantizationConfig const&, nvinfer1::builder::EngineBuildContext const&, nvinfer1::Network const&) ()
#5  0x000055557a8857b6 in nvinfer1::builder::Builder::buildInternal(nvinfer1::NetworkBuildConfig&, nvinfer1::NetworkQuantizationConfig const&, nvinfer1::builder::EngineBuildContext const&, nvinfer1::Network const&) ()
#6  0x000055557a886888 in nvinfer1::builder::Builder::buildEngineWithConfig(nvinfer1::INetworkDefinition&, nvinfer1::IBuilderConfig&) ()
#7  0x000055557a883d9b in main (argc=2, argv=0x7fffffffe468) at main.cc:133
```
