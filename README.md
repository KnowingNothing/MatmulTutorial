# TensorOp Matmul Tutorial

> This is an example repo for CUDA MatMul implementation. The aim of this repo is to provide some insights in high-performance kernel design for CUDA beginners. Currently, I only provide some implementation examples in `examples/matmul`.
Contributions for more kernels and other MatMul implementations are highly welcomed.

## About
There is a detailed [explanation](https://zhuanlan.zhihu.com/p/631227862) about the different versions of MatMul kernels in `examples/matmul`.

## Contents
- `examples`:

    `matmul`: The MatMul implementations

    `atom`: The usage of single intrinsic/instructions

## Plan
### More kernels
I plan to implement kernels for other operators such as softmax in future.

### Use CUTLASS in implementation
There is a plan to use the CuTe interface of CUTLASS to implement high-performance kernels.