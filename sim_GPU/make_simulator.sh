mkdir build
cd build
cmake -DTensorRT_DIR=/usr/src/tensorrt/ .. 
cmake -DCMAKE_PREFIX_PATH=$HOME/software/libtorch ..
make -j8
