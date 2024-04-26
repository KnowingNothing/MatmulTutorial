from dataclasses import dataclass
import subprocess
import os


WARP_SIZE = 32
WARP_GROUP_SIZE = 128
WARP_PER_WARP_GROUP = 4
MAX_CLUSTER_SIZE = 16


@dataclass
class dim3:
    x: int
    y: int
    z: int


@dataclass
class DeviceCoord:
    gridDim: dim3
    blockDim: dim3
    clusterDim: dim3
    blockIdx: dim3 = dim3(0, 0, 0)
    threadIdx: dim3 = dim3(0, 0, 0)
    blockIdx_in_cluster: dim3 = dim3(0, 0, 0)

    def block_id_in_cluster(self):
        return (
            self.blockIdx_in_cluster.x,
            self.blockIdx_in_cluster.y,
            self.blockIdx_in_cluster.z,
        )

    def set_blockIdx(self, x, y, z):
        self.blockIdx = dim3(x, y, z)
        self.blockIdx_in_cluster = dim3(x % self.clusterDim.x, y % self.clusterDim.y, z % self.clusterDim.z)

    def set_threadIdx(self, x, y, z):
        self.threadIdx = dim3(x, y, z)


@dataclass
class KernelHardwareInfo:
    device_id: int = 0
    sm_count: int = 0

    @staticmethod
    def query_device_multiprocessor_count(device_id: int = 0, arch: str = "90a"):
        cuda_header_code = f"""
#include <cuda_runtime.h>
#include <iostream>
static constexpr int device_id = {device_id};
    """
        cuda_code = """
int main() {
    cudaError_t result = cudaSetDevice(device_id);
    if (result != cudaSuccess) {
        std::cerr << "cudaSetDevice() returned error "
            << cudaGetErrorString(result) << std::endl;
        return 1;
    }
    int multiprocessor_count;
    result = cudaDeviceGetAttribute(&multiprocessor_count,
                                    cudaDevAttrMultiProcessorCount, device_id);
    if (result != cudaSuccess) {
        std::cerr << "cudaDeviceGetAttribute() returned error "
            << cudaGetErrorString(result) << std::endl;
        return 1;
    }
    std::cout << multiprocessor_count << std::endl;
    return 0;
}
    """
        # Combine the header and main CUDA code
        full_cuda_code = cuda_header_code + cuda_code

        # Write the CUDA code to a temporary file
        with open("temp_query_device.cu", "w") as file:
            file.write(full_cuda_code)

        # Compile the CUDA code using nvcc
        compile_command = (
            f"nvcc -arch=sm_{arch} temp_query_device.cu -o temp_query_device"
        )
        try:
            subprocess.run(
                compile_command,
                check=True,
                shell=True,
                text=True,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e.stderr}")
            return -1

        # Run the compiled binary and capture the output
        try:
            result = subprocess.run(
                "./temp_query_device", capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Execution failed: {e.stderr}")
            return -1
        finally:
            # Cleanup the temporary files
            os.remove("temp_query_device.cu")
            os.remove("temp_query_device")


if __name__ == "__main__":
    print(KernelHardwareInfo.query_device_multiprocessor_count())
