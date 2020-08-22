build:
	nvcc -w -lineinfo -std=c++11 simulator.cu -o test -lcublas
		./test simulator_input.txt aux_trace.txt

profile:
	nvcc -w -lineinfo -std=c++11 -arch=sm_70 main.cu -o test -lcublas -Xptxas="-v"