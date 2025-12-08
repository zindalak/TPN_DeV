#!/bin/bash --login
#SBATCH --ntasks=1       # number of CPUs
#SBATCH --mem-per-cpu=64G # memory for CPUs
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name hybrid_example

# module --force purge
# export PYTHONPATH="/mnt/scratch/baburish/doublepulse/env/doublepulse/lib/python3.11/site-packages/:$PYTHONPATH"

# eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`

# /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/Ubuntu_22.04_x86_64/metaprojects/icetray/v1.9.2/env-shell.sh

python /mnt/home/baburish/jax/TriplePandelReco_JAX/convert_pickle_to_tfrecords.py | tee /mnt/home/baburish/jax/TriplePandelReco_JAX/tfrecordsconversion-log.txt
