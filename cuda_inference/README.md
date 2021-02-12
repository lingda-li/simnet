## Building simulator

Run the `make` command to build the simulator.

Usuage:
 ./simulator <trace> <aux trace>
 
 ### Models supported
 Currently, two models `CNN3` and `CNN3_P` are supported. To select the CNN3 model, `CNN3_MODEL` flag should be defined in the models.cuh file. Else, CNN3_P model is used.
