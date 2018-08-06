import xarray as xr
import argparse
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input File Directory")
    parser.add_argument("-o", "--output", help="Output file directory")
    parser.add_argument("-x", "--xsub", type=int, default=2, help="X and Y subset factor")
    parser.add_argument("-z", "--zsub", type=int, default=1, help="Z subset factor")
    parser.add_argument("-t", "--tsub", type=int, default=1, help="Time subset factor")
    args = parser.parse_args()
    nc_files = sorted(glob(args.input + "*.nc"))
    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        ds.close()

if __name__ == "__main__":
    main()