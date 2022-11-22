#!/bin/bash
#DEBUG_FLAGS="-fPIC -g -fimplicit-none  -Wall  -O3 -Wline-truncation  -Wcharacter-truncation  -Wsurprising  -Waliasing  -Wimplicit-interface  -Wunused-parameter  -fwhole-file  -fcheck=all  -std=f2008  -pedantic  -fbacktrace -fbounds-check -ffpe-trap=zero,invalid,overflow,underflow"
DEBUG_FLAGS="-fPIC -O3 -pg"
F_INC="-I$NCAR_ROOT_INTEL/include -I$NCAR_INC_NETCDF -I$NCAR_INC_MKL"
F_LIB="-L$NCAR_ROOT_INTEL/lib -L$NCAR_LDFLAGS_NETCDF -L$NCAR_LDFLAGS_MKL -L$NCAR_LDFLAGS_MKLAUX"
all_paths="$F_INC $F_LIB $NCAR_LIBS_NETCDF -mkl"
rm *.mod *.o test_emulator
echo $FC $DEBUG_FLAGS -c module_neural_net.f90 tau_neural_net_quantile.f90 $all_paths
$FC $DEBUG_FLAGS -c module_neural_net.f90 tau_neural_net_quantile.f90 $all_paths
$FC $DEBUG_FLAGS test_quantile_emulator.f90 tau_neural_net_quantile.o module_neural_net.o -o test_quantile_emulator $all_paths
