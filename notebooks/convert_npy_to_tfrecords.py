#!/usr/bin/env python

import glob
import os
import copy
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import sys
sys.path.insert(0, "/mnt/home/baburish/jax/TriplePandelReco_JAX/")
from tfrecord_utils import serialize_example
import multiprocessing
import random
import gc

from typing import List, Dict, Any

# disable GPU
os.environ["JAX_PLATFORMS"] = "cpu"
tf.config.set_visible_devices([], 'GPU')

#indir = "/home/storage/hans/datasets/phototable/linear_fine/npy/"
#infiles_x = sorted(glob.glob(os.path.join(indir, "x_train*.npy")))
#infiles_y = sorted(glob.glob(os.path.join(indir, "y_train*.npy")))
#outdir = "/home/storage/hans/datasets/phototable/linear_fine/tfrecords/"
## this dataset has in total:
## 240203037 bins across 176 files (500m cut)
##98294012
# n_shards = 471
# n_bins_per_shard = 509985
# cartesian = True
## 240202935 after all shards are combined. loosing only 2 bins.
#n_input_files = 176

indir = "/mnt/research/IceCube/Gupta-Reco/photon_tables_npy/"
infiles_x = sorted(glob.glob(os.path.join(indir, "x_train*.npy")))
infiles_y = sorted(glob.glob(os.path.join(indir, "y_train*.npy")))
outdir = "/mnt/research/IceCube/Gupta-Reco/photon_tables_tfrecords/"
## this dataset has in total:
## 98294012 bins across 176 files (500m cut)

#This dataset has in total:
## 106208940 bins across 192 files
## 15599728
n_shards = 460
n_bins_per_shard = 230889
cartesian = True
n_input_files = 192

# shuffle
infiles = list(zip(infiles_x, infiles_y))
random.shuffle(infiles)
infiles_x, infiles_y = zip(*infiles)

infiles_xl = [[inf] for inf in infiles_x]
infiles_yl = [[inf] for inf in infiles_y]

dataset = tf.data.Dataset.from_tensor_slices((infiles_xl, infiles_yl))

def generate_and_write_a_shard(arguments: Dict[str, Any]) -> None:
    i: int = arguments['shard_index']
    n_bins_per_shard: int = arguments['n_bins_per_shard']

    x_t = arguments['x_train']
    y_t = arguments['y_train']

    compression_type = ''
    options = tf.io.TFRecordOptions(compression_type=compression_type)

    n = 0
    write_path = os.path.join(arguments['outdir'], f"data_shard_{i}.tfrecords")
    with tf.io.TFRecordWriter(write_path, options) as writer:
        for _, (x_instance, y_instance) in enumerate(zip(x_t, y_t)):
            writer.write(serialize_example(tf.constant(x_instance, dtype=tf.float32),
                                           tf.constant(y_instance, dtype=tf.float32))
                                        )
            n+=1

    print(f"Done! Wrote {n} events to tfrecord.")
    print(x_t.shape[0])
    print(f"Stored at {write_path}.")

def read_file(infile_x, infile_y):
    dat_x = tf.py_function(
        func=lambda path: np.load(path.numpy()[0].decode("utf-8")),
        inp=[infile_x],
        Tout=tf.float32
    )

    dat_y = tf.py_function(
        func=lambda path: np.load(path.numpy()[0].decode("utf-8")),
        inp=[infile_y],
        Tout=tf.float32
    )
    return (dat_x, dat_y)

dataset = dataset.map(read_file, num_parallel_calls=tf.data.AUTOTUNE)

n_total = 0
n_written = 0
x_train = []
y_train = []
shuffle_freq = 5
shard_idx = 0

for i, (x ,y) in enumerate(dataset):
    print(i, "current state:", sum([_tmp.shape[0] for _tmp in x_train]))

    x = x.numpy()
    y = y.numpy()

    # scale inputs fall within [-1, 1]
    km_scale = 1000
    if cartesian:
        x[:, 0] /= km_scale # x relative to track
        x[:, 3] /= km_scale # z relative to track
        # no need to scale
        # unit vector direction of track
        # since unit vector components
        # are scaled to [-1, 1]

    else:
        x[:, 0] /= km_scale
        x[:, 1] /= km_scale
        # normalize radians
        angle_scale = 2 * np.pi
        x[:, 2] /= angle_scale
        x[:, 4] /= angle_scale
        # note: cos(zenith) at column index 3 is already within [-1, 1]

    idx = np.isfinite(y.sum(axis=1))

    # and select reasonable parameter range for now.
    if cartesian:
        idx1 = np.logical_and(x[:, 0] > 1 / km_scale, x[:, 0] < 500 / km_scale)
        idx2 = np.logical_and(x[:, 3] > -800 / km_scale, x[:, 3] < 800 / km_scale)
        idx3 = np.logical_and(idx1, idx2)

    else:
        idx1 = np.logical_and(x[:, 0] > 1 / km_scale, x[:, 0] < 500 / km_scale)
        idx2 = np.logical_and(x[:, 1] > -800 / km_scale, x[:, 1] < 800 / km_scale)
        idx3 = np.logical_and(idx1, idx2)

    idx = np.logical_and(idx, idx3)
    n_total += x[idx].shape[0]
    print(x.shape[0], x[idx].shape[0])
    x_train.append(copy.copy(x[idx]))
    y_train.append(copy.copy(y[idx]))
    # memory_in_bytes = sys.getsizeof(x_train)
    # print(f"The variable '{x_train}' uses {memory_in_bytes} bytes of memory.")
    if (i+1) % shuffle_freq != 0 and i != n_input_files-1:
        # continue to accumulate
        continue

    # write
    x = np.concatenate(x_train)
    y = np.concatenate(y_train)

    x_train = []
    y_train = []

    x, y = shuffle(x, y)
    n_write_shards = x.shape[0] // n_bins_per_shard
    n_max = n_write_shards * n_bins_per_shard
    if n_max < x.shape[0]:
        x_train.append(copy.copy(x[n_max:]))
        y_train.append(copy.copy(y[n_max:]))

    arguments = []
    print(f"n_write_shards = {n_write_shards}")
    for m in range(n_write_shards):
        print(f"Appending")
        arguments.append(
            {
                'shard_index': shard_idx,
                'x_train': copy.copy(x[m*n_bins_per_shard : (m+1)*n_bins_per_shard]),
                'y_train': copy.copy(y[m*n_bins_per_shard : (m+1)*n_bins_per_shard]),
                'outdir': outdir,
                'n_bins_per_shard': n_bins_per_shard
            }
        )
        shard_idx += 1

    count = 1

    pool = multiprocessing.Pool(processes=count)
    pool.map(generate_and_write_a_shard, arguments)
    pool.close()
    pool.join()
    gc.collect()
    n_written += n_max

print("total seen: ", n_total)
print("total written", n_written)
print(len(x_train))
print("n_lost: ", x_train[0].shape)
print("n written", n_written)