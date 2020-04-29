#!/bin/bash
DEBUG_FLAGS="-fPIC -g -fimplicit-none  -Wall  -O3 -Wline-truncation  -Wcharacter-truncation  -Wsurprising  -Waliasing  -Wimplicit-interface  -Wunused-parameter  -fwhole-file  -fcheck=all  -std=f2008  -pedantic  -fbacktrace -fbounds-check -ffpe-trap=zero,invalid,overflow,underflow"
F_INC="-I$NCAR_INC_GNU -I$NCAR_INC_NETCDF -I$NCAR_INC_OPENBLAS"
F_LIB="-L$NCAR_LDFLAGS_GNU -L$NCAR_LDFLAGS_NETCDF -L$NCAR_LDFLAGS_OPENBLAS"
all_paths="$F_INC $F_LIB $NCAR_LIBS_NETCDF $NCAR_LIBS_OPENBLAS"
rm *.mod *.o test_emulator
$FC $DEBUG_FLAGS -c module_neural_net.f90 tau_neural_net_batch.f90 $all_paths
$FC $DEBUG_FLAGS -c neuralnet.f90 tau_neural_net.f90 $all_paths
$FC $DEBUG_FLAGS test_emulator.f90 tau_neural_net_batch.o module_neural_net.o -o test_emulator $all_paths
$FC $DEBUG_FLAGS test_emulator_old.f90 tau_neural_net.o neuralnet.o -o test_emulator_old $all_paths
