%% This script generates the mex files
% This folde contains some mex files for the acceleration of some routines.
% The compile commands are tuned for a Linux machine with GCC. If you use
% something different, you'll have to fix it for your case.

mex -R2018a CFLAGS='$CFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp' COPTIMFLAGS='$COPTIMFLAGS -fopenmp -O3' LDOPTIMFLAGS='$LDOPTIMFLAGS -fopenmp -O3' DEFINES='$DEFINES -fopenmp' assemble_hessian.c