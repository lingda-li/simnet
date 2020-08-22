## Building the simulator
This folder contains the script for building the simulator. Libtorch library is requried to build the simulator. 
Follow this article to get familier with libtorch installation and usuage. 
https://pytorch.org/cppdocs/installing.html
Please ensure you have the same version of libtorch installed as your pytorch version.
Provide the path to libtorch library in **make_simulator.sh** file.
Create a directory `build` in same folder. 
run `./make_simulator.sh` to build the simulator.


If the simulator builds normally, you will see 100% build message. If you get some error, please verify your libtorch version or your path to libtorch is correct.

### Parallel simulator
To build the parallel simulator, copy the contents of CMakeLists_parallel.txt to CMakeLists.txt or rename the file. Then use `./make_simulator.sh` to build the parallel simulator.
Usage: `./simulator <trace> <aux trace> <lat module> <# parallel traces>`
  Here, # parallel traces represents the number of parallel traces with each independent ROB or batch size of input.  
