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

python /mnt/home/baburish/jax/TriplePandelReco_JAX/extract_data_from_i3files/convert_i3_ftr_coinc_muonlabel.py -id /mnt/research/IceCube/Gupta-Reco/22646/0005000-0005999/ --dataset_id 22646 -s 5000 -e 6000 -o /mnt/research/IceCube/Gupta-Reco/22646/0005000-0005999-tfrecords/ --infile_base FinalLevel_NuMu_NuGenCCNC.022646 | tee /mnt/home/baburish/jax/TriplePandelReco_JAX/jobscripts/logs-22646/job6log.txt


