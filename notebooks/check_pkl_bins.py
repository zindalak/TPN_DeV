#!/usr/bin/env python

import numpy as np
import pickle
import os
import glob
from typing import Dict, List, Any
import sys
sys.path.insert(0, "/mnt/home/baburish/jax/TriplePandelReco_JAX/")

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


def load_table_from_pickle(infile: str) -> List[Any]:
    table = pickle.load(open(infile, "rb"))
    bin_info = dict()
    bin_info['dist'] = {'c': table['bin_centers'][0],
            'e': table['bin_edges'][0],
            'w': table['bin_widths'][0]}

    bin_info['rho'] = {'c': table['bin_centers'][1],
            'e': table['bin_edges'][1],
            'w': table['bin_widths'][1]}

    bin_info['z'] = {'c': table['bin_centers'][2],
            'e': table['bin_edges'][2],
            'w': table['bin_widths'][2]}

    bin_info['dt'] = {'c': table['bin_centers'][3],
            'e': table['bin_edges'][3],
            'w': table['bin_widths'][3]}
    # print(len(table['bin_centers'][0]))
    # print(len(table['bin_centers'][1]))
    # print(len(table['bin_centers'][2]))
    # print(len(table['bin_centers'][3]))
    # print(len(table['bin_centers'][3])+len(table['bin_centers'][2])+len(table['bin_centers'][1])+len(table['bin_centers'][0]))
    # print(len(bin_info['dt']['c'])+len(bin_info['dt']['e'])+len(bin_info['dt']['w']))
    # print(len(bin_info['dist']['c'])+len(bin_info['dist']['e'])+len(bin_info['dist']['w']))
    # print(len(bin_info['rho']['c'])+len(bin_info['rho']['e'])+len(bin_info['rho']['w']))
    # print(len(bin_info['z']['c'])+len(bin_info['z']['e'])+len(bin_info['z']['w']))
    print(len(table['bin_centers'][0]))
    return table, bin_info

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

print(f"Processing file {args.INFILE}")
table, bin_info = load_table_from_pickle(args.INFILE)
arguments = [dist for dist in bin_info['dist']['c']]
# print(f"table values = {table}")
# print(f"bin values = {bin_info}")
print(len(table), len(bin_info))