# debug for correctness
# baseline
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v00.cu main.cu -o test && ./test

# double buffer shared memory
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v01.cu main.cu -o test && ./test stages 2

# test performance
nvcc -arch=sm_80  matmul-v00.cu main.cu -o test && ./test