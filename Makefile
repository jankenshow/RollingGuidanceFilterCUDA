CUDA_VERSION="12.3"
CUDA_PATH="/usr/local/${CUDA_VERSION}"
CFLAGS="-I/usr/local/${CUDA_VERSION}/include"
LDFLAGS="-L/usr/local/${CUDA_VERSION}/lib64"
BUILD_PYTHON_BINDING=OFF

build: cmake
	make -C build 

cmake: mkdir
	CUDA_PATH=$(CUDA_PATH) CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) VERBOSE=1 cmake -B build -DBUILD_PYTHON_BINDING=$(BUILD_PYTHON_BINDING) -Dpybind11_DIR=`python -c 'import pybind11; print(pybind11.get_cmake_dir())'` .

mkdir: clean
	mkdir build

clean:
	rm -rf build .cache

python_test:
	python test/python/test_rgf.py
	python test/python/test_gauss.py