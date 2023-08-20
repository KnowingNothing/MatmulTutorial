```sh
git submodule update --init --recursive
cd ~/MatmulTutorial/3rdparty/tvm
mkdir build && cd build
cp ../cmake/config.cmake .
```

Edit config.cmake, find `USE_CUDA`, change it to `ON`. Make sure CUDA is in your PATH (e.g., `/usr/local/cuda`).
Find `USE_CUDNN`, `USE_CUBLAS`, and `USE_CUTLASS` change them to `ON`. Make sure you have installed CUDNN.

If you want to run TensorIR test file, you need to install LLVM. In the config.cmake, find `USE_LLVM` and change `OFF` to `path/to/llvm-config`.

```sh
cmake ..
make -j32
pip install --user numpy decorator attrs tornado psutil 'xgboost>=1.1.0' cloudpickle tqdm
cd ~/MatmulTutorial/examples/matmul/tvm
source env.sh
```

```sh
export CUDA_VISIBLE_DEVICES=0 # change the dev_id according to your settings
python relay_matmul_cublas.py # this takes minutes to finish
python relay_matmul_cutlass.py # this takes minutes to finish
```