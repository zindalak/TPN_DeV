import sys, os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--path_to_repo", type=str,
                  default="/home/storage/hans/jax_reco_gupta_corrections4/",
                  dest="PATH_TO_REPO",
                  help="directory containing the reco code")

parser.add_argument("-f", "--file_path", type=str,
                  default="/home/storage2/hans/i3files/21217/ftr/",
                  dest="PATH_TO_INPUT",
                  help="directory containing the event data .tfrecords files")

parser.add_argument("-tf", "--tfrecords_file", type=str,
                  default="/home/storage2/hans/i3files/21217/ftr/data_ds_21217_from_*_to_*_1st_pulse.tfrecord",
                  dest="TFRECORDS_FILE_NAME",
                  help="Name of the .tfrecords files containing event meta data")

parser.add_argument("-g", "--gpu", type=int,
                  default=0,
                  dest="GPU_INDEX",
                  help="which GPU should run the code")

parser.add_argument("-b", "--batch_size", type=float,
                  default=0.5,
                  dest="BATCH_SIZE",
                  help="how many events should go into one batch")
### This is non-trivial to interpret:
### The batch size is actually variable. Events with less doms will
### have larger batch size than events with many doms.
### In some sense this is just an overall scaling factor.
### See https://github.com/HansN87/TriplePandelReco_JAX/blob/e3692febe6050c483df3ad7b23a7d31f360c617f/lib/simdata_i3.py#L209
### If something crashes because of OOM (gpu out of memory), then choose a smaller value.

parser.add_argument("-n", "--network", type=str,
                  default="gupta_4comp_reg",
                  dest="NETWORK",
                  help="options are: gupta_4comp_reg, gupta_4comp")

parser.add_argument("-s", "--seed", type=str,
                    default="spline_mpe",
                    dest="SEED",
                    help="options are: spline_mpe, truth")

parser.add_argument("-nb", "--stop_after_n_batches", type=int,
                    default=100000000000,
                    dest="STOP_AFTER_N_BATCHES",
                    help="Set a small number if you want to test the script on a couple of batches")

parser.add_argument("-o", "--outfile", type=str,
                    default="results.npy",
                    dest="OUTFILE",
                    help="Where to write the reconstruction results")

# whether or not to shift the seed such that the vertex
# corresponds to the charge weighted median time of the event
parser.add_argument('--center_track_seed', default=True, action=argparse.BooleanOptionalAction)

# whether or not to use multiple vertex seeds: ~factor of 6 slower
parser.add_argument('--use_multiple_vertex_seeds', default=True, action=argparse.BooleanOptionalAction)

# whether or not to pre-scan the time axis to best-match the seed vertex.
parser.add_argument('--prescan_time', default=True, action=argparse.BooleanOptionalAction)


args = parser.parse_args()
print(args)
print("")

# Make code available to python
sys.path.insert(0, args.PATH_TO_REPO)

# Specify correct gpu
os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.GPU_INDEX}'

# Import JAX and require double precision.
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
dtype = jnp.float64

# Other tools.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import pandas as pd

# Import TriplePandel stuff
from lib.simdata_i3 import I3SimHandler
from lib.geo import center_track_pos_and_time_based_on_data
from lib.gupta_network_eqx_4comp import get_network_eval_v_fn
from lib.experimental_methods import get_vertex_seeds
from lib.simdata_i3 import I3SimBatchHandlerTFRecord
from lib.geo import center_track_pos_and_time_based_on_data_batched_v
from fitting.llh_fitter import get_fitter
from dom_track_eval import get_eval_network_doms_and_track
from likelihood_conv_mpe_padded_input_logsumexp_gupta import get_neg_c_triple_gamma_llh

# Assume 4-component gupta by default
n_hidden = 96
gupta = True
n_comp = 4

if args.NETWORK == "gupta_4comp_reg":
    network_path = os.path.join(args.PATH_TO_REPO, 'data/gupta/n96_4comp_w_penalty_1.e-4/new_model_no_penalties_tree_start_epoch_1000.eqx')

elif args.NETWORK == "gupta_4comp":
    network_path = os.path.join(args.PATH_TO_REPO, 'data/gupta/n96_4comp/new_model_no_penalties_tree_start_epoch_800.eqx')

else:
    raise NotImplementedError(f"network {args.NETWORK} not implemnted.")

# Network logic.
eval_network_v = get_network_eval_v_fn(bpath=network_path, dtype=dtype, n_hidden=n_hidden)
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v, dtype=dtype, gupta=gupta, n_comp=n_comp)

# Setup likelihood with 3 ns gaussian convolution.
neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track, sigma=3.0)

# Potential for additional stability via prescanning optimal vertex time
fit_llh = get_fitter(
                        neg_llh,
                        use_multiple_vertex_seeds=args.use_multiple_vertex_seeds,
                        prescan_time=args.prescan_time,
                        use_batches=True
                    )

# Load event input data in tfrecords format for efficient
# batched processing.
if '*' in args.TFRECORDS_FILE_NAME:
    fs = glob.glob(os.path.join(args.PATH_TO_INPUT, args.TFRECORDS_FILE_NAME))
    batch_maker = I3SimBatchHandlerTFRecord(fs, batch_size=args.BATCH_SIZE)

else:
    batch_maker = I3SimBatchHandlerTFRecord(fs, batch_size=args.BATCH_SIZE)

# Create padded batches (with different seq length).
batch_iter = batch_maker.get_batch_iterator()

# Get one batch.
# Note: You are free to extend over more batches.

collect_results = []
finished_batches = False
i = 0
while not finished_batches:
    if i == args.STOP_AFTER_N_BATCHES:
        print(f"Stopping early per user request (--stop_after_n_batches {args.STOP_AFTER_N_BATCHES}). Hence, did not reconstruct all available batches.")
        break

    try:
        print("reconstructing batch", i)
        pulse_data, meta_data = batch_iter.next() # [Nev, Ndom, Nobs], [Nev, Naux]

        pulse_data = jnp.array(pulse_data.numpy())
        meta_data = jnp.array(meta_data.numpy())
        # For the definition of different data fields / indices in meta_data
        # see https://github.com/HansN87/TriplePandelReco_JAX/blob/e3692febe6050c483df3ad7b23a7d31f360c617f/extract_data_from_i3files/convert_i3_tfrecord.py#L278C1-L282C74
        #        pulse_data = event_data[['x', 'y','z','time', 'charge']].to_numpy()
        #        meta_data = meta[['muon_energy_at_detector', 'q_tot', 'muon_zenith', 'muon_azimuth', 'muon_time',
        #                      'muon_pos_x', 'muon_pos_y', 'muon_pos_z', 'spline_mpe_zenith',
        #                      'spline_mpe_azimuth', 'spline_mpe_time', 'spline_mpe_pos_x',
        #                      'spline_mpe_pos_y', 'spline_mpe_pos_z']].to_numpy()


        if args.SEED == "spline_mpe":
            # Use SplineMPE as a seed.
            seed_data = meta_data[:, 8:14]

        elif args.SEED == "truth":
            seed_data = meta_data[:, 2:8]

        else:
            raise ValueError(f"seed {args.SEED} not available. Use spline_mpe or truth")

        seed_data = jnp.array(seed_data)
        centered_track_pos, centered_track_time = seed_data[: ,3:], seed_data[:, 2]
        track_src = seed_data[:, :2]

        if args.center_track_seed:
            print("shifting seed vertex.")
            centered_track_pos, centered_track_time = center_track_pos_and_time_based_on_data_batched_v(pulse_data, seed_data)

        print("seed vertex of first event in batch:", centered_track_pos[0], "m")
        print("data shape: ", pulse_data.shape)

        # JIT for speed (given the current batch tensor dimensions)
        fit_llh_jit = jax.jit(fit_llh).lower(track_src, centered_track_pos, centered_track_time, pulse_data).compile()

        # Run the fit
        solution = fit_llh_jit(track_src, centered_track_pos, centered_track_time, pulse_data)
        logl, direction, vertex, time = solution

        # And collect results and auxiliary data to be serialized to disk
        # once all batches are finished.
        # todo: output a nicer pandas.DataFrame instead of
        # raw numpy array.
        out_data = jnp.concatenate(
            [
                meta_data,
                jnp.expand_dims(logl, axis=1),
                direction,
                vertex,
                jnp.expand_dims(time, axis=1)
            ],
            axis=1
        )

        collect_results.append(out_data)
        print("")
        i += 1

    except StopIteration:
        print("Finished the last batch.")
        finished_batches = True

results = jnp.concatenate(collect_results, axis=0)
print("")
print("logl values (best-fit) of events across all batches:")
print(results[:, 14])
print("shape of result:", results.shape)

# serialize to disk
np.save(args.OUTFILE, results)

