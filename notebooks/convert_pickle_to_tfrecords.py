#!/usr/bin/env python

import numpy as np
import pickle

import sys, os
sys.path.insert(0, "/mnt/home/baburish/jax/TriplePandelReco_JAX")

from helpers import load_table_from_pickle
from helpers import get_effective_binomial_photons, get_prob_vals

import multiprocessing
import tensorflow as tf

from tfrecord_utils import serialize_example
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-i", "--infile", type=str,
                  default="/mnt/research/IceCube/Gupta-Reco/photon_tables_pkl/phototable_infinitemuon_zenith90.0_azimuth258.75.pkl",
                  dest="INFILE",
                  help="input pickle files")

parser.add_argument("-z", "--zenith", type=float,
                  default=90.0,
                  dest="ZENITH",
                  help="zenith of muon track [deg]")

parser.add_argument("-a", "--azimuth", type=float,
                  default=258.75,
                  dest="AZIMUTH",
                  help="azimuth of muon track [deg]")

parser.add_argument("-gl", "--gpu_memory_limit", type=int,
                  default=2048,
                  dest="GPU_MEMORY_LIMIT",
                  help="choose gpu to run on.")

parser.add_argument("-gi", "--gpu_id", type=int,
                  default=0,
                  dest="GPU_ID",
                  help="choose gpu to run on.")

parser.add_argument("-o", "--outdir", type=str,
                  default="/mnt/research/IceCube/Gupta-Reco/photon_tables_tfrecords/",
                  dest="OUTDIR",
                  help="directory where to write output tfrecords files")

args = parser.parse_args()


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[args.GPU_ID], 'GPU')
tf.config.set_logical_device_configuration(physical_devices[args.GPU_ID],
		[tf.config.LogicalDeviceConfiguration(memory_limit=args.GPU_MEMORY_LIMIT)]
)

def get_train_data(dist):
    global table, bin_info
    nvars = 3 # dist, rho, z
    ntrain = len(bin_info['z']['c']) * len(bin_info['rho']['c'])
    x_train = np.ones((ntrain, 3))
    y_train = np.ones((ntrain, len(bin_info['dt']['c'])))

    row_index = 0
    for t_z in bin_info['z']['c']:
        for t_rho in bin_info['rho']['c']:
            x_train[row_index, :] = [dist, t_z, t_rho]
            # todo: double check int cast in computation of n_tot
            n_obs, _ = get_effective_binomial_photons(dist, t_rho, t_z, table,
                                              bin_info, error_scale=10,
                                              as_float=True)

            y_train[row_index, :] = n_obs
            row_index += 1

    return x_train, y_train

print(f"Loading data from {args.INFILE}")
table, bin_info = load_table_from_pickle(args.INFILE)
arguments = [dist for dist in bin_info['dist']['c']]

count = 2 # how many cores to use to extract data from clsim tables
pool = multiprocessing.Pool(processes=count)
results = pool.map(get_train_data, arguments)
x_train, y_train = zip(*results)

x_train = np.concatenate(x_train, axis=0)
# add zenith, azimuth
n = len(x_train)
x_train = np.column_stack([x_train,
                            np.ones(n) * np.cos(np.deg2rad(args.ZENITH)),
                            np.ones(n) * np.deg2rad(args.AZIMUTH)
                        ])

# scale inputs fall within [-1, 1]
# normalize to km
km_scale = 1000
x_train[:, 0] /= km_scale
x_train[:, 1] /= km_scale
# normalize radians
angle_scale = np.pi
x_train[:, 2] /= angle_scale
x_train[:, 4] /= angle_scale
# note: cos(zenith) at column index 3 is already within [-1, 1]

y_train = np.concatenate(y_train, axis=0)

# invalid bins are bins that contain no data.
# better avoid from train set
idx = np.isfinite(y_train.sum(axis=1))

# and select reasonable parameter range for now.
idx1 = np.logical_and(x_train[:, 0] > 5 / km_scale, x_train[:, 0] < 400 / km_scale)
idx2 = np.logical_and(x_train[:, 1] > -400 / km_scale, x_train[:, 1] < 400 / km_scale)
idx3 = np.logical_and(idx1, idx2)

idx = np.logical_and(idx, idx3)

print(x_train[idx].shape)
print(y_train[idx].shape)

# tfrecords loop
write_path = os.path.join(
          args.OUTDIR,
          f"train_data_zenith_{args.ZENITH}_azimuth_{args.AZIMUTH}.tfrecords"
      )

compression_type = ''
options = tf.io.TFRecordOptions(compression_type=compression_type)

n_tot = 0
with tf.io.TFRecordWriter(write_path, options) as writer:
        for i, (x_arr, y_arr) in enumerate(zip(x_train[idx], y_train[idx])):
            writer.write(serialize_example(
                            tf.constant(x_arr, dtype=tf.float32),
                            tf.constant(y_arr, dtype=tf.float32),
                        )
                    )
            n_tot = i

print(f"Done! Wrote {n_tot+1} events to tfrecord.")