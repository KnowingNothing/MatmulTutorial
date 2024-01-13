#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaError_t cudaStatus;

    // Get the currently active device
    cudaStatus = cudaGetDevice(&device);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaGetDevice failed!" << std::endl;
        return 1;
    }

    // Query the shared memory size per block
    int sharedMemPerSM, sharedMemPerBlock;
    cudaStatus = cudaDeviceGetAttribute(&sharedMemPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceGetAttribute failed!" << std::endl;
        return 1;
    }
    cudaStatus = cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceGetAttribute failed!" << std::endl;
        return 1;
    }

    std::cout << "Shared memory available per sm: " << sharedMemPerSM << " bytes" << std::endl;
    std::cout << "Shared memory available per block: " << sharedMemPerBlock << " bytes" << std::endl;
    return 0;
}