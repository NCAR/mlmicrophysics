import argparse
import xarray as xr
import numpy as np
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration yaml file")
    parser.add_argument("-p", "--proc", type=int, default=1, help="Number of processors")
    args = parser.parse_args()

    return

if __name__ == "__main__":
    main()