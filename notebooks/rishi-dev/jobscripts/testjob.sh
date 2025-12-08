#!/bin/bash --login
#SBATCH --ntasks=1       # number of CPUs
#SBATCH --mem-per-cpu=80G # memory for CPUs
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name hybrid_example

module --force purge
export PYTHONPATH="/mnt/scratch/baburish/doublepulse/env/doublepulse/lib/python3.11/site-packages/:$PYTHONPATH"

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`

/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/Ubuntu_22.04_x86_64/metaprojects/icetray/v1.9.2/env-shell.sh

python /mnt/home/baburish/jax/TriplePandelReco_JAX/extract_data_from_i3files/convert_i3_ftr_coinc_muonlabel.py -id /mnt/research/IceCube/Gupta-Reco/21217/ --dataset_id 21217 -s 0000 -e 1000 -o /mnt/research/IceCube/Gupta-Reco/21217/ --infile_base Level2_IC86.2016_NuMu 


