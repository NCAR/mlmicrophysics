#!/bin/bash
NETCDF=/glade/u/apps/ch/opt/netcdf/4.6.1/intel/17.0.1/
#NETCDF=/opt/local
DEBUG_FLAGS="-fPIC -g -fimplicit-none  -Wall  -O3 -Wline-truncation  -Wcharacter-truncation  -Wsurprising  -Waliasing  -Wimplicit-interface  -Wunused-parameter  -fwhole-file  -fcheck=all  -std=f2008  -pedantic  -fbacktrace -fbounds-check -ffpe-trap=zero,invalid,overflow,underflow"
NC_FLAGS="-I/opt/local/include/ -L/opt/local/lib -lnetcdff -llapack -lblas"
rm *.mod *.o test_emulator
FC="gfortran"
#$FC $DEBUG_FLAGS -c neuralnet.f90 tau_neural_net.f90 $NC_FLAGS
$FC $DEBUG_FLAGS -c module_neural_net.f90 tau_neural_net_batch.f90 $NC_FLAGS

$FC $DEBUG_FLAGS test_emulator.f90 tau_neural_net_batch.o module_neural_net.o -o test_emulator  $NC_FLAGS
