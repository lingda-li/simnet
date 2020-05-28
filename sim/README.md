## Building the simulator
This folder contains the script for building the simulator. Libtorch library is requried to build the simulator. 
Follow this article to get familier with libtorch installation and usuage. 
https://pytorch.org/cppdocs/installing.html
Please ensure you have the same version of libtorch installed as your pytorch version.
Provide the path to libtorch library in **make_simulator.sh** file.
Create a directory `build` in same folder. 
run `./make_simulator.sh` to build the simulator.


If the simulator builds normally, you will see 100% build message. If you get some error, please verify your libtorch version or your path to libtorch is correct.
