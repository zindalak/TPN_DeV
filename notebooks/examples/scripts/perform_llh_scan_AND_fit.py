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

parser.add_argument("-ns", "--n_splits", type=int,
                  default=50,
                  dest="N_SPLITS",
                  help="split grid into some number of sequential pieces, to avoid GPU memory limitations")

parser.add_argument("-n", "--network", type=str,
                  default="gupta_4comp_reg",
                  dest="NETWORK",
                  help="options are: gupta_4comp_reg, gupta_4comp, gupta_3comp, gamma_3comp, custom")

parser.add_argument("-s", "--seed", type=str,
                    default="spline_mpe",
                    dest="SEED",
                    help="options are: spline_mpe, truth")

parser.add_argument("-c", "--gaussian_convolution_width", type=float,
                  default=3.0,
                  dest="GAUS_CONV_WIDTH",
                  help="how wide the convolution should be")


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
from lib.gupta_network_eqx_4comp import get_network_eval_v_fn, get_network_eval_v_fn_f32
from lib.experimental_methods import get_vertex_seeds
from fitting.llh_scanner import get_scanner
from fitting.llh_fitter import get_fitter
from dom_track_eval import get_eval_network_doms_and_track
from likelihood_conv_mpe_logsumexp_gupta import get_neg_c_triple_gamma_llh

# A custom color scheme
from palettable.cubehelix import Cubehelix
cx = Cubehelix.make(start=0.3, rotation=-0.5, n=16, reverse=False, gamma=1.0,
     	max_light=1.0,max_sat=0.5, min_sat=1.4).get_mpl_colormap()

# Specify the grid.
dzen = 0.07 # rad
dazi = 0.07 # rad
n_eval = 50 # number of grid points per axes

# Assume 4-component gupta by default
n_hidden = 96
gupta = True
n_comp = 4

if args.NETWORK == "custom":
    network_path = '/mnt/scratch/baburish/TPN-training/gupta_mixture_jax/test_no_penalties_tree_start_epoch_35.eqx'

elif args.NETWORK == "gupta_4comp_reg":
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
try:
    ni = "f64"
    print("Running f64 model")
    eval_network_v = get_network_eval_v_fn(bpath=network_path, dtype=dtype, n_hidden=n_hidden)
except:
    ni = "f32"
    print("Running f32 model")
    eval_network_v = get_network_eval_v_fn_f32(bpath=network_path, dtype=dtype, n_hidden=n_hidden)

# eval_network_v = get_network_eval_v_fn(bpath=network_path, dtype=dtype, n_hidden=n_hidden)
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
print("true direction:", true_src)

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
print(fitting_event_data.shape)

# Setup likelihood.
neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track, sigma=args.GAUS_CONV_WIDTH)

# Potential for additional stability via prescanning optimal vertex time

# First determine the best-fit
fit_llh = get_fitter(
                        neg_llh,
                        use_multiple_vertex_seeds=args.use_multiple_vertex_seeds,
                        prescan_time=args.prescan_time
                    )

# JIT! We want it to be fast.
fit_llh_jit = jax.jit(fit_llh)

# Run the fit
solution = fit_llh_jit(track_src, centered_track_pos, centered_track_time, fitting_event_data)
best_logl, best_direction, best_vertex, best_time= solution

print("")
print("solution found.")
print(f"logl: {best_logl:.3f}")
print(f"direction: {np.rad2deg(best_direction)} deg")
print("")


# Set up scanner.
# It splits the grid into sub-grids that are processed sequentially.
# This can avoid OOM errors if gpu memory is insufficient for entire grid.
scan_llh = get_scanner(
                        neg_llh,
                        use_multiple_vertex_seeds=args.use_multiple_vertex_seeds,
                        prescan_time=args.prescan_time,
                        n_splits=args.N_SPLITS,
                        use_jit=True
                    )

zenith = jnp.linspace(true_src[0]-dzen, true_src[0]+dazi, n_eval)
azimuth = jnp.linspace(true_src[1]-dzen, true_src[1]+dazi, n_eval)
X, Y = jnp.meshgrid(zenith, azimuth)

print("running the scan.")
# Run the scan.
solution = scan_llh(X, Y, best_vertex, best_time, fitting_event_data)
# use below if you want to use original seed values (not best-fit values)
# as seed for vertex minimization during scan.
#solution = scan_llh(X, Y, centered_track_pos, centered_track_time, fitting_event_data)

sol_logl, sol_vertex, sol_time = solution
logls = sol_logl.reshape(X.shape)

# Plot.
fig, ax = plt.subplots()
min_logl = np.amin(logls)
delta_logl = logls - np.amin(logls)
pc = ax.pcolormesh(np.rad2deg(X), np.rad2deg(Y), delta_logl, vmin=0, vmax=np.min([100, 1.2*np.amax(delta_logl)]), shading='auto', cmap=cx)
cbar = fig.colorbar(pc)
cbar.ax.tick_params(labelsize=16)
cbar.ax.get_yaxis().labelpad = 5
cbar.set_label("-2$\\Delta$log $L_{MPE}$", fontsize=20)
cbar.outline.set_linewidth(1.5)

contours = [4.61]
ix1, ix2 = np.where(delta_logl==0)
ax.scatter(np.rad2deg([X[ix1, ix2]]), np.rad2deg([Y[ix1, ix2]]), s=50, marker='o', facecolors='none', edgecolors='khaki', zorder=100., label='grid min')
ct = plt.contour(np.rad2deg(X), np.rad2deg(Y), delta_logl, levels=contours, linestyles=['solid'], colors=['khaki'], linewidths=1.0)

ax.scatter(np.rad2deg(true_src[0]), np.rad2deg(true_src[1]), marker="*", color='red', label="truth", zorder=200)
ax.scatter(np.rad2deg(track_src[0]), np.rad2deg(track_src[1]), marker="x", color='lime', label="seed", zorder=200)
ax.scatter(np.rad2deg(best_direction[0]), np.rad2deg(best_direction[1]), marker="x", color="magenta", label="best-fit", zorder=200)

ax.set_xlabel("zenith [deg]", fontsize=16)
ax.set_ylabel("azimuth [deg]", fontsize=16)
ax.set_xlim(np.rad2deg([true_src[0]-dzen, true_src[0]+dzen]))
ax.set_ylim(np.rad2deg([true_src[1]-dazi, true_src[1]+dazi]))
ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=16)

plt.legend()
plt.tight_layout()
plt.savefig(f"scan_ev_{args.EVENT_INDEX}.png", dpi=300)
