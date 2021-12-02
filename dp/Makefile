all:
	g++ -O3 -std=c++11 0_buildROB.cpp -o buildROB
	g++ -O3 -std=c++11 0_buildQ.cpp -o buildQ
	g++ -O3 -std=c++11 0_build1121.cpp -o build1121
	g++ -O3 -std=c++11 0_buildML.cpp -o buildML
	g++ -O3 -std=c++11 0_buildCxt.cpp -o buildCxt
	g++ -O3 -std=c++11 8_buildSim.cpp -o buildSim
	g++ -O3 -std=c++11 8_buildQSim.cpp -o buildQSim
	g++ -O3 -std=c++11 -DAC_TRACE 8_buildQSim.cpp -o buildQSimAC

1121:
	g++ -O3 -std=c++11 0_build1121.cpp -o build1121

1121_dbg:
	g++ -O2 -g -std=c++11 0_build1121.cpp -o build1121

dbg:
	g++ -O0 -g -std=c++11 0_buildROB.cpp -o buildROB
	g++ -O0 -g -std=c++11 0_buildQ.cpp -o buildQ
	g++ -O0 -g -std=c++11 0_buildML.cpp -o buildML
	g++ -O0 -g -std=c++11 0_buildCxt.cpp -o buildCxt
	g++ -O0 -g -std=c++11 8_buildSim.cpp -o buildSim
	g++ -O0 -g -std=c++11 8_buildQSim.cpp -o buildQSim
	g++ -O0 -g -std=c++11 -DAC_TRACE 8_buildQSim.cpp -o buildQSimAC
