INCLUDEDIR=/opt/spack/opt/spack/linux-ubuntu24.04-skylake/gcc-13.2.0/ipopt-3.14.9-36mdu2tluqotemn3rgh35dkletbyw55v/include/coin-or/

ipoptsolver: ipoptsolver.cpp
	g++ ipoptsolver.cpp kemeny_nlp.cpp -o ipoptsolve -larmadillo -fopenmp -lipopt -I${INCLUDEDIR}

armatest: armatests.cpp
	g++ armatests.cpp -o armatest -larmadillo

clean:
	rm ipoptsolve
