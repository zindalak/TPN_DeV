import multiprocessing
import subprocess
import os
import glob

if __name__ == '__main__':
    arguments = []
    # script_path = '/mnt/home/baburish/jax/TriplePandelReco_JAX/convert_pkl_to_npy.py'
    script_path = '/mnt/home/baburish/jax/TriplePandelReco_JAX/check_pkl_bins.py'
    indir = f"/mnt/research/IceCube/Gupta-Reco/photon_tables_pkl/"
    outdir = f"/mnt/research/IceCube/Gupta-Reco/photon_tables_npy/"
    cartesian = True

    infiles = glob.glob(os.path.join(indir, "*.pkl"))

    # collect arguments for each process
    for infile in infiles:
        argument_dict = dict()

        # extract zenith and azimuth from Matti's file names
        zenith = float(infile.split("_")[-2][6:])
        azimuth = float(infile.split("_")[-1][7:-4])

        argument_dict['zenith'] = zenith
        argument_dict['azimuth'] = azimuth
        argument_dict['infile'] = infile
        argument_dict['outdir'] = outdir

        arguments.append(argument_dict)
        cmd = ['python']
        cmd.append(f"{script_path}")

        for key in argument_dict.keys():
            val = argument_dict[key]
            cmd.append("--"+key)
            cmd.append(f"{val}")

        if cartesian:
            cmd.append("--cartesian")

        subprocess.run(cmd, shell=False)