# simnet
Simnet is a machine learning based hardware simulator to predict the latency of instructions. 

### Requirements:
    
    1. Python
    2. Pytorch
    3. CUDA (for GPU support)
    4. Libtorch 

### Folders
dp:     
    contains scripts for data pre-processing and running the simulator.

ml: 
    contains scripts for machine learning training & testing

sim: 
    contains scripts for building the simulator. 

data: 
    link to the data folder

Each folder contains the instruction.


### Building and running the simulator
SimNet requires instruction traces from gem5 for training and simulation.
The modified gem5 can be downloaded using

`git clone -b simnet https://github.com/lingda-li/gem5.git`

It generates a full instruction trace and a store queue trace when simulating a program execution.

#### Step 1: Train the machine learning model. 
The first step is to train a model which can predict latency of a given instruction. 

##### Preparing training data
Follow the instructions in section 1 (Preprocessing training data) `dp` folder to prepare the training data. 

1. Generate a.qq: `./dp/buildQ a.txt a.sq.txt`
2. Deduplicate qq file: `python dp/1_unique.py a.qq`
3. Make npy file: `python dp/2_Q-to-mmap.py --total-rows=<entry num> a.qqu`

##### Training model
Follow the instructions inside the `ml` folder for training the model. The trained module would be used by simulator.

#### Step 2: Build the simulator
Follow the instruction in `sim` folder to build the simulator. 

#### Step 3: Running for simulator
For running the simulator, we nedd to first process the instruction trace from gem5 to the simulator format. Follow the instructions in section 2 (Building simulator input) `dp` folder for processing input for simulator.
Place all the required files and `simulator` file build in step 2 in the same directory.
Note: Use the model converted to the libtorch version.

`./simulator_qq <trace> <aux trace> <lat module> <class module> <variances (optional)>`

or 

`./simulator_qq <trace> <aux trace> <lat module> <variances (optional)>`

Change log:
050820: Init the repo.
051420: Fix the simulator bug that didn't generate context instruction
relationship at runtime.
082820, v1.0: a stable version that works with datasets from O3CPU output that
includes instruction, register, data access, instruction access, and page table
walking information.
