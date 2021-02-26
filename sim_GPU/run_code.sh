mkdir build
cd build
cmake -DTensorRT_DIR=/usr/src/tensorrt/ ..
make -j8
