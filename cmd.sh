# debug for correctness
# cublas
nvcc -arch=sm_80 call_cublas.cu -lcublas -o test_cublas && ./test_cublas

# baseline
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v00.cu main.cu -o test && ./test

# multi-stage
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v01.cu main.cu -o test && ./test stages 4
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v02.cu main.cu -o test && ./test stages 4
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v03.cu main.cu -o test && ./test stages 4
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v04.cu main.cu -o test && ./test stages 4
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v05.cu main.cu -o test && ./test stages 4
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v06.cu main.cu -o test && ./test stages 4
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v07.cu main.cu -o test && ./test stages 4
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v08.cu main.cu -o test && ./test stages 4 multi_threading 2
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v09.cu main.cu -o test && ./test stages 4 multi_threading 2
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v10.cu main.cu -o test && ./test stages 4 multi_threading 2
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v11.cu main.cu -o test && ./test stages 4 multi_threading 2
nvcc -arch=sm_80  -DDEBUG -Xcompiler -fopenmp matmul-v12.cu main.cu -o test && ./test stages 4

# test performance
nvcc -arch=sm_80  matmul-v00.cu main.cu -o test && ./test

# multi-stage
nvcc -arch=sm_80  matmul-v01.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_80  matmul-v02.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_80  matmul-v03.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_80  matmul-v04.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_80  matmul-v05.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_80  matmul-v06.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_80  matmul-v07.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_80  matmul-v08.cu main.cu -o test && ./test stages 4 multi_threading 2 iters 200
nvcc -arch=sm_80  matmul-v09.cu main.cu -o test && ./test stages 4 multi_threading 2 iters 200
nvcc -arch=sm_80  matmul-v10.cu main.cu -o test && ./test stages 4 multi_threading 2 iters 200
nvcc -arch=sm_80  matmul-v11.cu main.cu -o test && ./test stages 4 multi_threading 2 iters 200
nvcc -arch=sm_80  matmul-v12.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_80  matmul-v13.cu main.cu -o test && ./test stages 4 iters 200

nvcc -arch=sm_86  matmul-v01.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_86  matmul-v02.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_86  matmul-v03.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_86  matmul-v04.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_86  matmul-v05.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_86  matmul-v06.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_86  matmul-v07.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_86  matmul-v08.cu main.cu -o test && ./test stages 4 multi_threading 2
nvcc -arch=sm_86  matmul-v09.cu main.cu -o test && ./test stages 4 multi_threading 2
nvcc -arch=sm_86  matmul-v10.cu main.cu -o test && ./test stages 4 multi_threading 2
nvcc -arch=sm_86  matmul-v11.cu main.cu -o test && ./test stages 4 multi_threading 2 iters 200
nvcc -arch=sm_86  matmul-v12.cu main.cu -o test && ./test stages 4 iters 200
nvcc -arch=sm_86  matmul-v13.cu main.cu -o test && ./test stages 4 iters 200