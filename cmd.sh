# debug for correctness
# cublas
nvcc -arch=sm_80 call_cublas.cu -lcublas -o test_cublas && ./test_cublas

# baseline
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v00.cu main.cu -o test && ./test

# multi-stage
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v01.cu main.cu -o test && ./test stages 4

# test performance
nvcc -arch=sm_80  matmul-v00.cu main.cu -o test && ./test

# multi-stage
nvcc -arch=sm_80  matmul-v01.cu main.cu -o test && ./test stages 4