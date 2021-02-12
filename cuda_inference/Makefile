build:
	nvcc -w -lineinfo -std=c++11 -arch=sm_70 simulator.cu -o test -lcublas
		./test simulator_input.txt aux_trace.txt

parallel:
	nvcc -w -lineinfo -std=c++11 -arch=sm_70 parall_simnet.cu -o test -lcublas
		./test simulator_input.txt aux_trace.txt 1

profile:
	nvcc -w -lineinfo -std=c++11 -arch=sm_70 main.cu -o test -lcublas -Xptxas="-v"