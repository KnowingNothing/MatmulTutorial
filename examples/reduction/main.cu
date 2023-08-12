#include "helper.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

template <typename T> extern void reduce(T *A, T *B, int length);

template <class T>
void callReduce(int length, int iterations = 100, bool check = true) {
  unsigned int bytes = length * sizeof(T);
  T *hPtr = (T *)malloc(bytes);
  for (int i = 0; i < length; ++i) {
    hPtr[i] = (T)(rand() & 0xFF);
  }

  T *hOutPtr = (T *)malloc(sizeof(T));

  T *dPtr;

  CUDA_CHECK(cudaMalloc((void **)&dPtr, bytes));
  CUDA_CHECK(cudaMemcpy(dPtr, hPtr, bytes, cudaMemcpyHostToDevice));

  // get results
  reduce<T>(dPtr, hOutPtr, length);
  CUDA_CHECK(cudaGetLastError());

  if (check) {
    T sum = hPtr[0];
    for (int i = 1; i < length; ++i) {
      sum += hPtr[i];
    }
    std::cout << "CPU result: " << sum << "\n";
    std::cout << "GPU result: " << hOutPtr[0] << "\n";
  }

  // warm-up
  reduce<T>(dPtr, hOutPtr, length);

  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start);
  cudaDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    reduce<T>(dPtr, hOutPtr, length);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Running cost of CUDA kernel is "
            << duration.count() / 1e3 / iterations << "ms\n";
  std::cout << "Throughput(GB/s): "
            << (float)length * sizeof(T) /
                   ((float)duration.count() / 1e3 / iterations) * 1e3 / 1e9
            << "\n";
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float ms;
  // cudaEventElapsedTime(&ms, start, stop);
  // std::cout << "Running cost of CUDA kernel is " << double(ms) / iterations
  // << "ms\n"; std::cout << "Throughput(GB/s): " << (float)length * sizeof(T) /
  // (double(ms) / iterations) * 1e3 / 1e9 << "\n";

  free(hPtr);
  free(hOutPtr);
  CUDA_CHECK(cudaFree(dPtr));
}

int main(int argc, char **argv) {
  std::cout << argv[0] << " Starting...\n";
  int length = 1024;
  int devId = 0;
  int iterations = 100;
  bool check = true;

  if (argc > 1) {
    assert(argc % 2 == 1 && "Should specify key-value in arguments.");
    for (int i = 1; i < argc; i += 2) {
      std::string key(argv[i]);
      char *value = argv[i + 1];
      if (key.size() && key[0] == '-') {
        key = key.substr(1);
      }
      if (key == "len" || key == "length") {
        length = std::atoi(value);
      } else if (key == "dev" || key == "device") {
        devId = std::atoi(value);
      } else if (key == "iter" || key == "iterations") {
        iterations = std::atoi(value);
      } else if (key == "check") {
        std::string checkStr(value);
        if (checkStr == "true") {
          check = true;
        } else if (checkStr == "false") {
          check = false;
        } else {
          std::cerr << "Unknow check arg: " << checkStr << "\n";
          return 1;
        }
      } else {
        std::cerr << "Unknown arg: " << key << "\n";
        return 1;
      }
    }
  }

  std::cout << "length = " << length << "\n";
  std::cout << "devId = " << devId << "\n";
  std::cout << "iterations = " << iterations << "\n";

  CUDA_CHECK(cudaSetDevice(devId));

  callReduce<float>(length, iterations, check);
  return 0;
}