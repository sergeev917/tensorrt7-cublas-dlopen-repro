# syntax = docker/dockerfile:1.0-experimental
FROM ubuntu:18.04 as builder
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq -o=Dpkg::Use-Pty=0 update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        g++-8 \
        vim \
        wget
WORKDIR /w
RUN --mount=type=bind,source=vendor,target=/w/vendor \
    sha1sum --strict --check vendor/checksums && \
    mkdir -p /w/include /w/lib /w/lib.dyn /w/debs && \
    dpkg-deb --fsys-tarfile vendor/cuda-repo-ubuntu1604-11-1-local_11.1.1-455.32.00-1_amd64.deb | \
    tar --directory=/w/debs --strip-components=3 --wildcards --verbose \
        --extract './var/cuda-repo-ubuntu1604-11-1-local/*.deb' && \
    dpkg-deb --fsys-tarfile debs/cuda-cudart-dev-11-1_11.1.74-1_amd64.deb | \
    tar --directory=/w/lib --strip-components=7 --wildcards --verbose \
        --extract './usr/local/cuda-11.1/targets/x86_64-linux/lib/*.a' && \
    dpkg-deb --fsys-tarfile debs/libcublas-dev-11-1_11.3.0.106-1_amd64.deb | \
    tar --directory=/w/lib --strip-components=7 --wildcards --verbose \
        --extract './usr/local/cuda-11.1/targets/x86_64-linux/lib/*.a' && \
    dpkg-deb --fsys-tarfile debs/cuda-nvrtc-dev-11-1_11.1.105-1_amd64.deb | \
    tar --directory=/w/lib --strip-components=8 --wildcards --verbose \
        --extract './usr/local/cuda-11.1/targets/x86_64-linux/lib/stubs/*.so' && \
    dpkg-deb --fsys-tarfile debs/cuda-driver-dev-11-1_11.1.74-1_amd64.deb | \
    tar --directory=/w/lib --strip-components=8 --wildcards --verbose \
        --extract './usr/local/cuda-11.1/targets/x86_64-linux/lib/stubs/*.so' && \
    tar --directory=/w/lib --strip-components=2 --wildcards --verbose \
        --extract 'cuda/lib64/*.a' --gzip < vendor/cudnn-11.1-linux-x64-v8.0.5.39.tgz && \
    tar --directory=/w/lib --strip-components=4 --wildcards --verbose \
        --extract 'TensorRT-7.2.1.6/targets/x86_64-linux-gnu/lib/*.a' \
        --gzip < vendor/TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz && \
    tar --directory=/w/include --strip-components=2 --wildcards --verbose \
        --extract 'TensorRT-7.2.1.6/include/*' \
        --gzip < vendor/TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz && \
    rm -rf image debs
RUN wget "https://media.githubusercontent.com/media/onnx/models/f884b33c3e2371952aad7ea091898f418c830fe5/vision/classification/squeezenet/model/squeezenet1.0-3.onnx"
COPY main.cc /w/
RUN g++ -O2 -ggdb3 \
    -Wno-deprecated-declarations \
    -o main \
    main.cc \
    -I ./include \
    ./lib/libnvinfer_static.a \
    ./lib/libmyelin_compiler_static.a \
    ./lib/libmyelin_pattern_library_static.a \
    ./lib/libmyelin_executor_static.a \
    ./lib/libmyelin_pattern_runtime_static.a \
    ./lib/libnvonnxparser_static.a \
    ./lib/libprotobuf.a \
    ./lib/libonnx_proto.a \
    ./lib/libnvinfer_plugin_static.a \
    ./lib/libcudnn_static.a \
    ./lib/libcublas_static.a \
    ./lib/libcublasLt_static.a \
    ./lib/libcudart_static.a \
    ./lib/libculibos.a \
    ./lib/libcudadevrt.a \
    -L lib \
    -lcuda \
    -lnvrtc \
    -lrt \
    -ldl \
    -pthread

FROM nvidia/cuda:11.1-runtime-ubuntu18.04
# we've linked with libcublas_static.a and libcublasLt_static.a above,
# no need in dynamic cublas libraries
RUN apt purge -y libcublas-11-1
WORKDIR /w
COPY --from=builder /w/main /w/squeezenet1.0-3.onnx /w/
ENTRYPOINT ["/w/main", "./squeezenet1.0-3.onnx"]
