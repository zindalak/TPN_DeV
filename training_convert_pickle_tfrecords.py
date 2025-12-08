"""
This script repeatedly calls convert_i3_ftr.py to extract event information (e.g. pulses)
from some number of .i3 files into a single feather file.
Repeated calls are processed in parallel on multiple cores.
"""

import multiprocessing
import subprocess
import os
import glob


def start_process(argument_dict):
    script_path = '/mnt/home/baburish/jax/TriplePandelReco_JAX/convert_pickle_to_tfrecords.py'
    cmd = ['python']
    cmd.append(f"{script_path}")

    for key in argument_dict.keys():
        val = argument_dict[key]
        cmd.append("--"+key)
        cmd.append(f"{val}")

    return subprocess.run(cmd, shell=False)


if __name__ == '__main__':
    #count = multiprocessing.cpu_count() // 2
    count = 2
    pool = multiprocessing.Pool(processes=count)

    arguments = []
    indir = f"/mnt/research/IceCube/Gupta-Reco/photon_tables_pkl/"
    outdir = f"/mnt/research/IceCube/Gupta-Reco/photon_tables_tfrecords/"

    infiles = glob.glob(os.path.join(indir, "*.pkl"))

    # collect arguments for each process
    for infile in infiles:
        argument_dict = dict()

        # extract zenith and azimuth from Matti's file names
        zenith = float(infile.split("_")[-2][6:])
        azimuth = float(infile.split("_")[-1][7:-4])
        file = "train_data_zenith_"+str(zenith)+"_azimuth_"+str(azimuth)+".tfrecords"
        file_path = outdir+file
        if os.path.exists(file_path):
            print(f"File {file} exits")
            continue
        argument_dict['zenith'] = zenith
        argument_dict['azimuth'] = azimuth
        argument_dict['infile'] = infile
        argument_dict['outdir'] = outdir

        arguments.append(argument_dict)
    # start processes
    pool.map(start_process, arguments)