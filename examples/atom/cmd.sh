# singl-wmma
nvcc -arch=sm_80 single-wmma.cu -o single-wmma
nvcc -arch=sm_80 -ptx single-wmma.cu -o single-wmma.ptx
cuobjdump --dump-sass single-wmma > single-wmma.sass