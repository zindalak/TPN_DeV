import sys, os
sys.path.insert(0, "/home/storage/hans/jax_reco_new")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tensorflow_probability.substrates import jax as tfp

import jax.numpy as jnp
from jax.scipy import optimize
import jax
jax.config.update("jax_enable_x64", True)
import optimistix as optx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lib.simdata_i3 import I3SimBatchHandlerTFRecord
from lib.geo import center_track_pos_and_time_based_on_data_batched_v
from lib.smaller_network import get_network_eval_v_fn
from lib.experimental_methods import get_vertex_seeds

from likelihood_conv_mpe_padded_input_w_noise import get_neg_c_triple_gamma_llh
from lib.geo import get_xyz_from_zenith_azimuth, __c
from dom_track_eval import get_eval_network_doms_and_track
import time

dtype = jnp.float64
eval_network_v = get_network_eval_v_fn(bpath='/home/storage/hans/jax_reco_new/data/smaller_network',
                                       dtype=dtype)
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v, dtype=dtype)


event_ids = ['1022', '10393', '10644', '10738', '11086', '11232', '13011',
       '13945', '14017', '14230', '15243', '16416', '16443', '1663',
       '1722', '17475', '18846', '19455', '20027', '21113', '21663',
       '22232', '22510', '22617', '23574', '23638', '23862', '24530',
       '24726', '25181', '25596', '25632', '27063', '27188', '27285',
       '28188', '28400', '29040', '29707', '3062', '31920', '31989',
       '32781', '32839', '33119', '33656', '34506', '35349', '37086',
       '37263', '37448', '37786', '37811', '39166', '39962', '40023',
       '41381', '41586', '42566', '42568', '42677', '43153', '43483',
       '4397', '44081', '48309', '48448', '48632', '49067', '50832',
       '51687', '51956', '54374', '55301', '55526', '55533', '56041',
       '5620', '56741', '56774', '57174', '57394', '57723', '59010',
       '59029', '59089', '59099', '59228', '62274', '62512', '63373',
       '65472', '6586', '8', '8604', '8674', '8840', '9410', '9419',
       '9505']

# set up likelihood for batched processing
neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track)

scale = 10.0
scale_rad = 100.0

@jax.jit
def neg_llh_5D(x, args):
        centered_track_time = args[0]
        fitting_event_data = args[1]

        # project back if outside of [0, pi] x [0, 2*pi]
        zenith = x[0] / scale_rad
        azimuth = x[1] / scale_rad
        zenith = jnp.fmod(zenith, 2.0*jnp.pi)
        zenith = jnp.where(zenith < 0, zenith+2.0*jnp.pi, zenith)
        cond = zenith > jnp.pi
        zenith = jnp.where(cond, -1.0*zenith+2.0*jnp.pi, zenith)
        azimuth = jnp.where(cond, azimuth-jnp.pi, azimuth)

        azimuth = jnp.fmod(azimuth, 2.0*jnp.pi)
        azimuth = jnp.where(azimuth < 0, azimuth+2.0*jnp.pi, azimuth)

        projected_dir = jnp.array([zenith, azimuth])
        return neg_llh(projected_dir, x[2:]*scale, centered_track_time, fitting_event_data)

# optimization logic for one seed.
def optimize_one_seed(x0, data, centered_track_time):
    args=[centered_track_time, data]
    solver = optx.BFGS(rtol=1e-8, atol=1e-4, use_inverse=True)
    best_x = optx.minimise(neg_llh_5D, solver, x0, args=args, throw=False).value
    best_logl = neg_llh_5D(best_x, args=args)
    return best_logl, best_x

# generalize to multiple seeds
optimize_multiple_seeds = jax.jit(jax.vmap(optimize_one_seed, (0, None, None), (0, 0)))

# optimization logic for one event.
def optimize_one_event(data, track_src, centered_track_time, centered_track_pos):
    # generate multiple seeds
    vertex_seeds = get_vertex_seeds(centered_track_pos, track_src)
    track_src = jnp.expand_dims(track_src, axis=0)
    track_src = jnp.repeat(track_src, vertex_seeds.shape[0], axis=0)
    x0 = jnp.column_stack([track_src*scale_rad, vertex_seeds/scale])

    logls, x = optimize_multiple_seeds(x0, data, centered_track_time)
    logls = jnp.where(jnp.isnan(logls), jnp.nanmax(logls), logls)

    idx = jnp.argmin(logls)
    best_logl = logls[idx]
    best_x = x[idx]

    return best_logl, best_x

# make it work on a batch.
optimize_one_batch = jax.jit(jax.vmap(optimize_one_event, (0, 0, 0, 0), (0, 0)))

# make logl calculation work on a batch.
neg_llh_one_batch = jax.jit(jax.vmap(neg_llh, (0, 0, 0, 0), 0))

def reconstruct_one_batch(data, mctruth):
    mctruth_ = mctruth[:, 2:8]
    centered_track_positions, centered_track_times = \
            center_track_pos_and_time_based_on_data_batched_v(data, mctruth_)

    track_src_v = mctruth_[:, :2]

    true_logl = neg_llh_one_batch(track_src_v,
                                  centered_track_positions,
                                  centered_track_times,
                                  data)

    result_logl, result_x = optimize_one_batch(data,
                                           track_src_v,
                                           centered_track_times,
                                           centered_track_positions)

    return true_logl - result_logl, result_x


for event_id in event_ids:
    print("working on event:", event_id)
    bp = '/home/storage2/hans/i3files/alerts/bfrv2/filter_prepulse/'
    tfrecord = os.path.join(bp, f'event_{event_id}_N100_from_0_to_10_1st_pulse.tfrecord')

    batch_size = 100 # let's load all events in one go to determine what's the max number of doms hit across all events.
    n_bins = 1
    batch_maker = I3SimBatchHandlerTFRecord(tfrecord,
                                            n_bins=n_bins,
                                            bucket_batch_sizes=[batch_size] * (n_bins+2),
                                            pad_to_bucket_boundary=False)

    # the events will have been padded to the same length as the one with the highest n_doms.
    batch_iter = batch_maker.get_batch_iterator()
    try:
        data, mctruth = batch_iter.next()
    except:
        print("failed reading event.")
        continue

    n_doms_max = data.shape[1]
    print(n_doms_max)

    # reload batch iterator with correct max_doms value
    # but smaller batch size. This avoids recompilation between batches.
    batch_size = 20
    batch_maker = I3SimBatchHandlerTFRecord(tfrecord,
                                            n_bins=n_bins,
                                            bucket_batch_sizes=[batch_size] * (n_bins+2),
											n_doms_max = n_doms_max+1,
                                            pad_to_bucket_boundary=True)

    batch_iter = batch_maker.get_batch_iterator()
    results = []
    for i in range(5):
        try:
            data, mctruth = batch_iter.next()
        except:
            print("failed reading batch.")
            continue

        data, mctruth = (jnp.array(data), jnp.array(mctruth))
        print(f"processing batch with shape ({data.shape[0]}, {data.shape[1]}, {data.shape[2]})")
        print(mctruth.shape)

        tic = time.time()
        delta_logl, result_x = reconstruct_one_batch(data, mctruth)
        result_x.block_until_ready()
        toc = time.time()

        y = jnp.column_stack([mctruth, result_x, delta_logl])
        print(f"took {toc-tic:.1f}s.")
        results.append(y)

    # store results.
    results = jnp.concatenate(results, axis=0)
    np.save(f"reco_result_{event_id}_filter_prepulse_multiple_seeds_tfrecord.npy", results)







