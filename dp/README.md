## Preprocessing Steps
---
Before starting the model training, we need to pre-process the input 
Required Input: Instruction trace i.e. output of gem5

### Required package for preprocessing
Python: 
    1. numpy
    2. sortedcontainers
### Steps:
---
First, we need to generate the required executables with command `make`. 
#### Step 1: Build the Re-order buffer (ROB)
Run the executable and dump the output to a new file one_out.txt.

`./0_buildROB 500.perlbench_r.10.txt > one_out.txt`

#### Step 2: Remove the duplicates
Remove the duplicate entries and shuffle the entries. This will output a file with name **one_out.txtu**.

`python 1_unique.py one_out.txt`

#### Step 3: Conversion
Convert variables to numpy format. This will output a *npz* file.

`python 2_ml-to-npy.py one_out.txtu `

#### Step 4: Input normalizeation  
Normalzie the input with mean and variance. This will output multiple file. The stats of the input data i.e. mean and variance will be stored in the stats folder. The normalized file will be in the same directory with normalize tag i.e. **normalize_one_out.txtu**.

`python 3_scale.py --save one_out.txtu.t0.npz`

The terminal shows the number of features, global mean and global variance.

#### Step 5: Combine
Combine all the normalized npz file (if multiple). Output totall.npz

`python 4_combine.py normall_one_out.txtu.t0.npz`

#### Step 6: Test and train dataset
Divide the datasets for test and training

`python 6_get_test.py totalall.npz`

### Other Scripts
---
`5_stats.py`: Displays values seen for different features of the dataset in a sorted list. The input for this script is the output from the `1_unique.py` i.e. **one_out.txtu**.

`python 5_stats.py one_out.txtu`

The stats shown are:

1. Context length
2. Time out
3. Time in
4. Instruction type
5. \# of source reginsters
6. Source register type
7. Source register index
8.  \# of dest registers
9. Destination register type
10. Destination register index
11. Register
12. PC


