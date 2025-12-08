import sys, os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--path_to_repo", type=str,
                  default="/home/storage/hans/jax_reco_gupta_corrections4/",
                  dest="PATH_TO_REPO",
                  help="directory containing the reco code")

parser.add_argument("-f", "--file_path", type=str,
                  default="/home/fast_storage/i3/22645/ftr/",
                  dest="PATH_TO_INPUT",
                  help="directory containing the event data .ftr files")

parser.add_argument("-mf", "--meta_file", type=str,
                  default="meta_ds_22645_from_0_to_1000_10_to_100TeV.ftr",
                  dest="META_FILE_NAME",
                  help="Name of the .ftr file containing event meta data")

parser.add_argument("-pf", "--pulses_file", type=str,
                  default="pulses_ds_22645_from_0_to_1000_10_to_100TeV.ftr",
                  dest="PULSES_FILE_NAME",
                  help="Name of the .ftr  file containing event pulses data")

parser.add_argument("-g", "--gpu", type=int,
                  default=0,
                  dest="GPU_INDEX",
                  help="which GPU should run the code")

parser.add_argument("-e", "--event_index", type=int,
                  default=0,
                  dest="EVENT_INDEX",
                  help="Which event should be used. Index within input file.")

parser.add_argument("-n", "--network", type=str,
                  default="gupta_4comp_reg",
                  dest="NETWORK",
                  help="options are: gupta_4comp_reg, gupta_4comp, gupta_3comp, gamma_3comp")

parser.add_argument("-s", "--seed", type=str,
                    default="spline_mpe",
                    dest="SEED",
                    help="options are: spline_mpe, truth")

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

# Import TriplePandel stuff
from lib.simdata_i3 import I3SimHandler
from lib.geo import center_track_pos_and_time_based_on_data
from lib.gupta_network_eqx_4comp import get_network_eval_v_fn
from lib.experimental_methods import get_vertex_seeds
from fitting.llh_fitter import get_fitter
from dom_track_eval import get_eval_network_doms_and_track
from likelihood_conv_mpe_logsumexp_gupta import get_neg_c_triple_gamma_llh

# Assume 4-component gupta by default
n_hidden = 96
gupta = True
n_comp = 4

if args.NETWORK == "gupta_4comp_reg":
    network_path = os.path.join(args.PATH_TO_REPO, 'data/gupta/n96_4comp_w_penalty_1.e-4/new_model_no_penalties_tree_start_epoch_1000.eqx')

elif args.NETWORK == "gupta_4comp":
    network_path = os.path.join(args.PATH_TO_REPO, 'data/gupta/n96_4comp/new_model_no_penalties_tree_start_epoch_800.eqx')

elif args.NETWORK == "gupta_3comp":
    # a 3 component gupta needs a different import
    from lib.gupta_network_eqx import get_network_eval_v_fn
    n_comp = 3
    network_path = os.path.join(args.PATH_TO_REPO, 'data/gupta/n96_w_penalty_1.e-3/new_model_no_penalties_tree_start_epoch_260.eqx')

elif args.NETWORK == "gamma_3comp":
    # a 3 component gamma needs different imports
    from lib.small_network import get_network_eval_v_fn
    from likelihood_conv_mpe_w_noise_logsumexp import get_neg_c_triple_gamma_llh
    n_comp = 3
    gupta = False
    network_path = os.path.join(args.PATH_TO_REPO, 'data/small_network')

else:
    raise NotImplementedError(f"network {args.NETWORK} not implemnted.")

# Network logic.
eval_network_v = get_network_eval_v_fn(bpath=network_path, dtype=dtype, n_hidden=n_hidden)
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v, dtype=dtype, gupta=gupta, n_comp=n_comp)

# Get an IceCube event.
#bp = '/home/fast_storage/i3/22645/ftr/'
sim_handler = I3SimHandler(
					os.path.join(args.PATH_TO_INPUT, args.META_FILE_NAME),
                    os.path.join(args.PATH_TO_INPUT, args.PULSES_FILE_NAME),
                    os.path.join(args.PATH_TO_REPO, 'data/icecube/detector_geometry.csv')
				)

meta, pulses = sim_handler.get_event_data(args.EVENT_INDEX)
print(f"muon energy: {meta['muon_energy_at_detector']/1.e3:.1f} TeV")

# Get dom locations, first hit times, and total charges (for each dom).
event_data = sim_handler.get_per_dom_summary_from_sim_data(meta, pulses)

# Remove early pulses.
sim_handler.replace_early_pulse(event_data, pulses)
print("n_doms", len(event_data))

# Get MCTruth.
true_pos = jnp.array([meta['muon_pos_x'], meta['muon_pos_y'], meta['muon_pos_z']])
true_time = meta['muon_time']
true_zenith = meta['muon_zenith']
true_azimuth = meta['muon_azimuth']
true_src = jnp.array([true_zenith, true_azimuth])
print("true direction:", np.rad2deg(true_src), "deg")

if args.SEED == "spline_mpe":
    # Use SplineMPE as a seed.
    track_pos = jnp.array([meta['spline_mpe_pos_x'], meta['spline_mpe_pos_y'], meta['spline_mpe_pos_z']])
    track_time = meta['spline_mpe_time']
    track_zenith = meta['spline_mpe_zenith']
    track_azimuth = meta['spline_mpe_azimuth']
    track_src = jnp.array([track_zenith, track_azimuth])

elif args.SEED == "truth":
    track_pos = true_pos
    track_time = true_time
    track_zenith = true_zenith
    track_azimuth = true_azimuth
    track_src = true_src

else:
    raise ValueError(f"seed {args.SEED} not available. Use spline_mpe or truth")

print("seed direction:", np.rad2deg(track_src), "deg")
print("original seed vertex:", track_pos, "m")

centered_track_pos, centered_track_time = track_pos, track_time
if args.center_track_seed:
    print("shifting seed vertex.")
    centered_track_pos, centered_track_time = center_track_pos_and_time_based_on_data(event_data, track_pos, track_time, track_src)

print("seed vertex:", centered_track_pos, "m")

fitting_event_data = jnp.array(event_data[['x', 'y', 'z', 'time', 'charge']].to_numpy())
print("data shape: ", fitting_event_data.shape)

# Setup likelihood.
neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track)

# Potential for additional stability via prescanning optimal vertex time
fit_llh = get_fitter(
                        neg_llh,
                        use_multiple_vertex_seeds=args.use_multiple_vertex_seeds,
                        prescan_time=args.prescan_time
                    )

# JIT the thing. We want it to be fast.
fit_llh_jit = jax.jit(fit_llh)

# Run the fit
solution = fit_llh_jit(track_src, centered_track_pos, centered_track_time, fitting_event_data)
logl, direction, vertex, time = solution

print("")
print("solution found.")
print(f"logl: {logl:.3f}")
print(f"direction: {np.rad2deg(direction)} deg")
print(f"position: {vertex} m")
