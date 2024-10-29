INCLUDEDIR=/software/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-13.2.0/ipopt-3.14.9-ow3yd5466cnmagj5twm56zkdayasujyu/include/coin-or

ipoptsolver: ipoptsolver.cpp
	g++ ipoptsolver.cpp kemeny_nlp.cpp -o ipoptsolve -larmadillo -fopenmp -lipopt -I${INCLUDEDIR}

clean:
	rm ipoptsolve
