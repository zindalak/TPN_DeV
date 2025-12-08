#!/usr/bin/env python

import numpy as np
import pickle
import os
import glob

import sys
sys.path.insert(0, "/mnt/home/baburish/jax/TriplePandelReco_JAX/")

from helpers import load_table_from_pickle
from helpers import get_effective_binomial_photons, get_prob_vals

import multiprocessing

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-i", "--infile", type=str,
                  default="/mnt/research/IceCube/Gupta-Reco/photon_tables_pkl/phototable_infinitemuon_zenith90.0_azimuth258.75.pkl",
                  dest="INFILE",
                  help="input pickle files")

parser.add_argument("-z", "--zenith", type=float,
                  default=95.0,
                  dest="ZENITH",
                  help="zenith of muon track [deg]")

parser.add_argument("-a", "--azimuth", type=float,
                  default=-139.19690010631788,
                  dest="AZIMUTH",
                  help="azimuth of muon track [deg]")

parser.add_argument("-o", "--outdir", type=str,
                  default="/mnt/research/IceCube/Gupta-Reco/photon_tables_npy/",
                  dest="OUTDIR",
                  help="directory where to write output tfrecords files")

parser.add_argument("-c", "--cartesian", default=False,
                  action="store_true",
                  dest="CARTESIAN",
                  help="if angular coordinates should be encoded in cartesian coordinates")

args = parser.parse_args()


def transform_dimensions(dist, t_rho, t_z):
    x0 = dist
    x1 = np.cos(t_rho)
    x2 = np.sin(t_rho)
    x3 = t_z
    return [x0, x1, x2, x3]


def get_train_data(dist):
    global table, bin_info
    nvars = 4 # dist, rho, z
    ntrain = len(bin_info['z']['c']) * len(bin_info['rho']['c'])
    x_train = np.ones((ntrain, nvars))
    y_train = np.ones((ntrain, len(bin_info['dt']['c'])))

    row_index = 0
    for t_z in bin_info['z']['c']:
        for t_rho in bin_info['rho']['c']:
            if args.CARTESIAN:
                x_train[row_index, :] = transform_dimensions(dist, t_rho, t_z)
            else:
                x_train[row_index, :] = [dist, t_z, t_rho]
            # todo: double check int cast in computation of n_tot
            n_obs, _ = get_effective_binomial_photons(dist, t_rho, t_z, table,
                                              bin_info, error_scale=1,
                                              as_float=True)
            y_train[row_index, :] = n_obs
            row_index += 1

    return x_train, y_train


table, bin_info = load_table_from_pickle(args.INFILE)
arguments = [dist for dist in bin_info['dist']['c']]

count = 15
pool = multiprocessing.Pool(processes=count)
results = pool.map(get_train_data, arguments)

pool.close()
pool.join()

x_train, y_train = zip(*results)

x_train = np.concatenate(x_train, axis=0)

if args.CARTESIAN:
    x_train = np.column_stack([x_train,
                                np.ones(x_train.shape[0]) * np.cos(np.deg2rad(args.ZENITH)),
                                np.ones(x_train.shape[0]) * np.sin(np.deg2rad(args.ZENITH)) * np.cos(np.deg2rad(args.AZIMUTH)),
                                np.ones(x_train.shape[0]) * np.sin(np.deg2rad(args.ZENITH)) * np.sin(np.deg2rad(args.AZIMUTH))
                            ])
else:
    x_train = np.column_stack([x_train,
                                np.ones(x_train.shape[0]) * np.cos(np.deg2rad(args.ZENITH)),
                                np.ones(x_train.shape[0]) * np.deg2rad(args.AZIMUTH)
                            ])

y_train = np.concatenate(y_train, axis=0)

# invalid bins are bins that contain no data.
# better avoid
idx = np.isfinite(y_train.sum(axis=1))

print(x_train[idx].shape)
print(y_train[idx].shape)

np.save(os.path.join(args.OUTDIR, f"x_train_zenith_{args.ZENITH}_azimuth_{args.AZIMUTH}.npy"), x_train[idx])
np.save(os.path.join(args.OUTDIR, f"y_train_zenith_{args.ZENITH}_azimuth_{args.AZIMUTH}.npy"), y_train[idx])