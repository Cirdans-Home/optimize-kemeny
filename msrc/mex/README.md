# Mex files

This folder contains the implementation of some routines as mex files in C in order to speed up some calculations. The matlab file contains a command to compile the files in question that is specific to the use of gcc on a linux machine. If the setting is different it may be necessary to adjust the instructions.

| **File** | |
|------|-|
| assemble_hessian.c | Routine that uses OpenMP accelerations of some loops for the assembly of the Hessian matrix in the dense case. |
