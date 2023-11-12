#include "common.h"
#include "reference.h"

const int testM = 259;
const int testN = 253;
const int testK = 251;

int main(int argc, char** argv) {
    int M = testM;
    int N = testN;
    int K = testK;
    using AType = half;
    using BType = half;
    using CType = half;
    using AccumType = float;
    AccumType alpha = 0.9;
    AccumType beta = 0.1;

    std::vector<int> AShape = {M, K};
    std::vector<int> BShape = {N, K};
    std::vector<int> CShape = {M, N};
    auto hA = alloc_cpu_tensor<AType>(AShape);
    random_fill(hA, AShape);
    auto hB = alloc_cpu_tensor<BType>(BShape);
    random_fill(hB, BShape);
    auto hC = alloc_cpu_tensor<CType>(CShape);
    auto goldenC = alloc_cpu_tensor<CType>(CShape);
    auto dA = alloc_gpu_tensor<AType>(AShape);
    auto dB = alloc_gpu_tensor<BType>(BShape);
    auto dC = alloc_gpu_tensor<CType>(CShape);

    /// timers
    CPUTimer cpu_timer;
    GPUTimer gpu_timer;

    /// compute golden value
    std::cout << "Comupting cpu golden values...\n";
    GemmParams cpu_params{M, N, K, hA, hB, goldenC, alpha, beta};
    cpu_timer.tick();
    reference_cpu_gemm(cpu_params);
    cpu_timer.tick();
    std::cout << "CPU golden done! Use " << cpu_timer.report_last_ms() << " ms.\n";

    /// copy data
    std::cout << "Copying data from CPU to GPU...\n";
    cpu_timer.tick();
    copy_to_gpu(hA, dA, AShape);
    copy_to_gpu(hB, dB, BShape);
    cpu_timer.tick();
    std::cout << "Copy data done! Use " << cpu_timer.report_last_ms() << " ms.\n";

    /// compute gpu reference
    std::cout << "Computing gpu reference values...\n";
    GemmParams gpu_params(M, N, K, dA, dB, dC, alpha, beta);
    gpu_timer.sync_all();
    gpu_timer.tick();
    reference_gpu_gemm(gpu_params);
    gpu_timer.tick();
    gpu_timer.sync_all();
    std::cout << "GPU reference done! Use " << gpu_timer.report_last_ms() << " ms.\n";
    
    /// copy results
    std::cout << "Copying results...\n";
    copy_to_cpu(hC, dC, CShape);
    std::cout << "Copying results done!\n";

    /// compare results
    assert_allclose(hC, goldenC, CShape, /*rtol=*/1e-3);
    std::cout << "All done!";

    free_cpu_tensor(hA);
    free_cpu_tensor(hB);
    free_cpu_tensor(hC);
    free_cpu_tensor(goldenC);
    free_gpu_tensor(dA);
    free_gpu_tensor(dB);
    free_gpu_tensor(dC);
    return 0;
}