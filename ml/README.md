### Models
This folder contains scripts for traning, testing and converting the model to libtorch format. 
#### Required package
1. matplotlib
2. sklearn

#### Scripts
1. `train_cla.py`: Classification based prediction of latency, 
2. `train_lat.py`: Direct prediction of latency value.
3. `test_lat.py`: Test inference for the latency model.
4. `test_cla.py`: Test classification based prediction model.
5. `models.py`: Definition of all machine learning models. 

#### Current Models
| Name | Combination |
|------------|-------------|
| CNN3 |  |
| CNN_3F |  |
| CNN2_P |  |
| CNN3_P |  |
| CNN5 |  |
| CNN3_P2 |  |
| CNN3_P_lat |  |
| CNN3_FPB |  |
| CNN3_P_D_C |  |

#### Training the models 
We use the training data generated by data proprocessing in `dp` folder. Copy the `data_spec` folder here. `data_spec` folder should have two files: `totalall.npz` and `statsall.npz`. 

##### Step 1
Run `python train_cla.py` or `train_lat.py` as decribed above. You can define the training size and testing size from the script.

##### Step 2
Run `test_com.py` or `test_lat.py` for testing the trained dataset.

##### Step 3
Convert the trained model into libtorch format. The trained model is supposed to be in `..data_spec\models\`.

Run `python convert.py`. This script generates a libtorch version of trained model. Use the converted model with simulator. 