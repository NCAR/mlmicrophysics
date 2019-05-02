#!/bin/bash
NETCDF=/glade/u/apps/ch/opt/netcdf/4.6.1/intel/17.0.1/
rm *.mod *.o test_emulator
ifort -c -fPIC -g neuralnet.f90 tau_neural_net.f90 -I$NETCDF/include -L$NETCDF/lib
ifort test_emulator.f90 *.o -fPIC -g -o test_emulator
