#! /bin/bash

# Script to configure and compile Tensorflow on habanero

module load anaconda/3-4.4.0
source activate tensorflow

PYTHON_BIN_PATH=$(which python3.6) \
TF_NEED_JEMALLOC=1 \
TF_NEED_GCP=1 \
TF_NEED_S3=1 \
TF_NEED_HDFS=0 \
TF_NEED_KAFKA=0 \
TF_ENABLE_XLA=0 \
TF_NEED_GDR=0 \
TF_NEED_VERBS=0 \
TF_NEED_MPI=0 \
TF_NEED_OPENCL_SYCL=0 \
TF_NEED_CUDA=1 \
TF_CUDA_VERSION=9.0 \
TF_CUDNN_VERSION=7 \
CUDA_TOOLKIT_PATH=/cm/shared/apps/cuda90/toolkit/current/ \
CUDNN_INSTALL_PATH=$USER_DIR/opt/cudnn7+cuda9.0/ \
TF_CUDA_COMPUTE_CAPABILITIES=3.7,6.0 \
TF_CUDA_CLANG=0 \
TF_NEED_TENSORRT=0 \
TF_SET_ANDROID_WORKSPACE=0 \
GCC_HOST_COMPILER_PATH=$(which gcc) \
CC_OPT_FLAGS='-march=native' \
./configure

export TF_MKL_ROOT=$USER_DIR/opt/intel/mkltf
export LD_LIBRARY_PATH="$USER_DIR/opt/cudnn7+cuda9.0/lib64/:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$USER_DIR/opt/cuda90/lib64/stubs:$LD_LIBRARY_PATH"

bazel --batch --output_user_root $USER_DIR/bazel build --config=mkl --config=opt \
    --copt='-DEIGEN_USE_MKL_ALL' --copt='-DMKL_DIRECT_CALL' --verbose_failures \
    --action_env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package $USER_DIR/binaries/tensorflow-nightly/

