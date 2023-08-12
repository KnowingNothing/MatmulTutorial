#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(status)                                                \
{                                                                         \
    cudaError_t error = status;                                           \
    if (error != cudaSuccess)                                             \
    {                                                                     \
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                    << " at line: " << __LINE__ << std::endl;             \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
}
