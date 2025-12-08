import sys, os
sys.path.insert(0, "/home/storage/hans/jax_reco_gupta_corrections/")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tensorflow_probability.substrates import jax as tfp

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import optimistix as optx
import numpy as np

from lib.simdata_i3 import I3SimBatchHandlerTFRecord
from lib.geo import center_track_pos_and_time_based_on_data_batched_v
from lib.gupta_network_eqx import get_network_eval_v_fn
from lib.experimental_methods import get_vertex_seeds

from likelihood_conv_mpe_padded_input_w_noise_logsumexp_gupta import get_neg_c_triple_gamma_llh
from lib.geo import get_xyz_from_zenith_azimuth, __c
from dom_track_eval import get_eval_network_doms_and_track

import time

dtype = jnp.float64
eval_network_v = get_network_eval_v_fn(bpath='/home/storage/hans/github/data/gupta/n96_w_penalty_1.e-3/new_model_no_penalties_tree_start_epoch_260.eqx', dtype=dtype, n_hidden=96)
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v, dtype=dtype, gupta=True)


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

scale = 30.0
scale_rad = 100.0


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


def neg_llh_3D(x, args):
        track_dir = args[0]
        centered_track_time = args[1]
        fitting_event_data = args[2]
        return neg_llh(track_dir, x*scale, centered_track_time, fitting_event_data)


def optimize_one_seed(x0, data, centered_track_time, true_track_dir):
    # optimization logic for one seed.
    solver = optx.BFGS(rtol=1e-10, atol=1e-6, use_inverse=True)

    args=[centered_track_time, data]
    best_x_5D = optx.minimise(neg_llh_5D, solver, x0, args=args, throw=False).value
    best_logl_5D = neg_llh_5D(best_x_5D, args=args)

    args=[true_track_dir, centered_track_time, data]
    best_x_3D = optx.minimise(neg_llh_3D, solver, x0[2:], args=args, throw=False).value
    best_logl_3D = neg_llh_3D(best_x_3D, args=args)

    return best_logl_5D, best_logl_3D, best_x_5D, best_x_3D

# generalize to multiple seeds
optimize_multiple_seeds = jax.vmap(optimize_one_seed, (0, None, None, 0), (0, 0, 0, 0))


def optimize_one_event(data, track_src, centered_track_time, centered_track_pos):
    # optimization logic for one event.
    # generate multiple seeds
    vertex_seeds = get_vertex_seeds(centered_track_pos, track_src)
    track_src = jnp.expand_dims(track_src, axis=0)
    track_src = jnp.repeat(track_src, vertex_seeds.shape[0], axis=0)

    # perform fits
    x0 = jnp.column_stack([track_src*scale_rad, vertex_seeds/scale])
    logls_5D, logls_3D, x_5D, x_3D = optimize_multiple_seeds(x0, data, centered_track_time, track_src)

    # pick best result of 5D fits
    logls_5D = jnp.where(jnp.isnan(logls_5D), jnp.nanmax(logls_5D), logls_5D)
    idx = jnp.argmin(logls_5D)
    best_logl_5D = logls_5D[idx]
    best_x_5D = x_5D[idx]

    # pick best result of 3D fits
    logls_3D = jnp.where(jnp.isnan(logls_3D), jnp.nanmax(logls_3D), logls_3D)
    idx = jnp.argmin(logls_3D)
    best_logl_3D = logls_3D[idx]
    best_x_3D = x_3D[idx]

    return best_logl_5D, best_logl_3D, best_x_5D, best_x_3D

# make it work on a batch.
optimize_one_batch = jax.vmap(optimize_one_event, (0, 0, 0, 0), (0, 0, 0, 0))

# make logl calculation work on a batch.
neg_llh_one_batch = jax.vmap(neg_llh, (0, 0, 0, 0), 0)


def reconstruct_one_batch(data, meta_data):
    mctruth = meta_data[:, 2:8]
    # select 'muon_zenith', 'muon_azimuth', 'muon_time', 'muon_pos_x', 'muon_pos_y', 'muon_pos_z' as mctruth.
    # see https://github.com/HansN87/TriplePandelReco_JAX/blob/4732a7d97791199c286d587fd8ed7bc5d2d4baad/lib/simdata_i3.py#L105
    # for definition of indices if you need something else.
    centered_track_positions, centered_track_times = \
            center_track_pos_and_time_based_on_data_batched_v(data, mctruth)

    track_src_v = mctruth[:, :2]
    logl_at_mctruth = neg_llh_one_batch(track_src_v,
                                  centered_track_positions,
                                  centered_track_times,
                                  data)

    logl_5D, logl_3D, x_5D, x_3D = optimize_one_batch(data,
                                           track_src_v,
                                           centered_track_times,
                                           centered_track_positions)

    return jnp.concatenate(
            [
                jnp.expand_dims(logl_5D, axis=1),
                jnp.expand_dims(logl_3D, axis=1),
                jnp.expand_dims(logl_at_mctruth, axis=1),
                x_5D,
                x_3D
            ],
            axis=1
        )

# Let's finally do some work.
for event_id in event_ids:
    print("working on event:", event_id)
    bp = '/home/storage2/hans/i3files/alerts/bfrv2/NN_corrections/tfrecord/'
    tfrecord = os.path.join(bp, f'data_event_{event_id}_1st_pulse.tfrecord')
    if not os.path.exists(tfrecord):
        print("no correction for ", event_id, " skipping!")
        continue

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
    print("max doms:", n_doms_max)

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

    while True:
        try:
            dom_data, event_meta_data = batch_iter.next()
        except:
            print("Can not read another batch. Seems like we are done here.")
            break

        dom_data, event_meta_data = (jnp.array(dom_data), jnp.array(event_meta_data))
        print(np.sum(dom_data[:, :, 4]))
        print(f"processing batch with dom data shape ({dom_data.shape[0]}, {dom_data.shape[1]}, {dom_data.shape[2]})")
        print(f"and event meta data shape ({event_meta_data.shape[0]}, {event_meta_data.shape[1]})")

        tic = time.time()

        reconstruct_one_batch_jit = jax.jit(reconstruct_one_batch).lower(dom_data, event_meta_data).compile()
        result = reconstruct_one_batch_jit(dom_data, event_meta_data)
        result.block_until_ready()
        toc = time.time()

        y = jnp.column_stack([result, event_meta_data])
        print(f"took {toc-tic:.1f}s.")
        results.append(y)

    results = jnp.concatenate(results, axis=0)

    # store results according to giacommo's files.
    # notice we could store more stuff, like event meta information, the likelihood at the full MC truth,
    # or the position fitted in the 3D fit etc ...
    llh_5D = results[:, 0:1]
    llh_3D = results[:, 1:2]
    delta_llh = llh_3D - llh_5D
    llh_results = jnp.concatenate([delta_llh, llh_5D, llh_3D], axis=1)

    best_x = results[:, 3:5]

    np.save(f"results/corrected/llh_results_event_{event_id}_padded_input", llh_results)
    np.save(f"results/corrected/mincoords_event_{event_id}_padded_input", best_x/scale_rad)
    np.save(f"results/corrected/truecoords_event_{event_id}_padded_input", event_meta_data[:, 2:4])
    np.save(f"results/corrected/splinempecoords_event_{event_id}_padded_input", event_meta_data[:, 8:10])







