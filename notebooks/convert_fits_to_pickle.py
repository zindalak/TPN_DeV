#!/usr/bin/env python

from icecube.clsim.tablemaker.photonics import FITSTable
import pickle
import glob
import os

from typing import Tuple
import multiprocessing

# tracks
# fs = glob.glob('/mnt/research/IceCube/Gupta-Reco/photon_tables/*.fits')
# outdir = "/mnt/research/IceCube/Gupta-Reco/newholeice_photontables/"
fs = glob.glob('/mnt/research/IceCube/Gupta-Reco/newholeice_photontables/*.fits')
outdir = "/mnt/research/IceCube/Gupta-Reco/photon_tables_pkl"
# print(fs)

arguments = []
for f in fs:
    fname = f.split("/")[-1]
    fname = '.'.join(fname.split(".")[:-1])
    out_fname = os.path.join(outdir, fname+".pkl")
    arguments.append((f, out_fname))


def convert_files(paths: Tuple[str, str]) -> None:
    infile, outfile = paths
    print(f"Starting conversion: {os.path.basename(infile)}")
    table = FITSTable.load(infile)

    data = dict()
    data['bin_centers'] = table.bin_centers
    data['bin_widths'] = table.bin_widths
    data['bin_edges'] = table.bin_edges
    data['values'] = table.values
    data['weights'] = table.weights
    pickle.dump(data, open(outfile, "wb"))
    print(f"Completed conversion: {os.path.basename(outfile)}")
    return

#convert_files(arguments[0])

if __name__ == "__main__":
    count = 2
    with multiprocessing.Pool(processes=count) as pool:
        pool.map(convert_files, arguments)