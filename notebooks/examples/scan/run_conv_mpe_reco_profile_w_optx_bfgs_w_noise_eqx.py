#!/usr/bin/env python

import sys, os
sys.path.insert(0, "/home/storage/hans/jax_reco_gupta_corrections3/")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['XLA_FLAGS'] = "--xla_disable_hlo_passes=constant_folding"

from tensorflow_probability.substrates import jax as tfp

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import optimistix as optx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# TriplePandelSPE/JAX stuff
from lib.simdata_i3 import I3SimHandler
from lib.geo import center_track_pos_and_time_based_on_data
from lib.gupta_network_eqx_4comp import get_network_eval_v_fn
from dom_track_eval import get_eval_network_doms_and_track
from likelihood_conv_mpe_w_noise_logsumexp_gupta import get_neg_c_triple_gamma_llh
from palettable.cubehelix import Cubehelix
cx =Cubehelix.make(start=0.3, rotation=-0.5, n=16, reverse=False, gamma=1.0,
                           max_light=1.0,max_sat=0.5, min_sat=1.4).get_mpl_colormap()

import time

# Number of scan points on 1D
n_eval = 50 # making it a 20x20 grid

# Scan range (truth +/- dzen, +/- dazi)
#dzen = 0.2 # rad
#dazi = 0.2 # rad

dzen = 0.004
dazi = 0.004

# Event Index.
event_index = int(sys.argv[1])

# Get network and eval logic.
dtype = jnp.float64

# Split grid into sub-grids that are processed sequentially.
# This can avoid OOM errors if gpu memory is insufficient for entire grid.
n_splits = 25

eval_network_v = get_network_eval_v_fn(bpath='/home/storage/hans/jax_reco_gupta_corrections3/data/gupta/n96_4comp/new_model_no_penalties_tree_start_epoch_800.eqx', dtype=dtype, n_hidden=96)
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v, dtype=dtype, gupta=True, n_comp=4)

# Get an IceCube event.
bp = '/home/storage2/hans/i3files/alerts/ftp-v1_flat/energy_loss_network_inputs/npe/ftr/w_corrections/'

#event_id = 11086
event_id = 35349
#event_id = 20027
sim_handler = I3SimHandler(os.path.join(bp, f'meta_ds_data_event_{event_id}_w_NN_correction.ftr'),
                                os.path.join(bp, f'pulses_ds_data_event_{event_id}_w_NN_correction.ftr'),
                                '/home/storage/hans/jax_reco_new/data/icecube/detector_geometry.csv')

meta, pulses = sim_handler.get_event_data(event_index)
print(f"muon energy: {meta['muon_energy_at_detector']/1.e3:.1f} TeV")

# Get dom locations, first hit times, and total charges (for each dom).
event_data = sim_handler.get_per_dom_summary_from_sim_data(meta, pulses)
# Remove early pulses.
sim_handler.replace_early_pulse(event_data, pulses)
# now load data with charge corrections
clean_time = event_data['time']

event_data = sim_handler.get_per_dom_summary_from_sim_data(meta, pulses, correct_charge=True)
# and update with early pulse cleaned time
event_data['time'] = clean_time

print("n_doms", len(event_data))

# Make MCTruth seed.
track_pos = jnp.array([meta['muon_pos_x'], meta['muon_pos_y'], meta['muon_pos_z']])
track_time = meta['muon_time']
track_zenith = meta['muon_zenith']
track_azimuth = meta['muon_azimuth']
track_src = jnp.array([track_zenith, track_azimuth])

print("original seed vertex:", track_pos)
centered_track_pos, centered_track_time = center_track_pos_and_time_based_on_data(event_data, track_pos, track_time, track_src)
print("shifted seed vertex:", centered_track_pos)

centered_track_time = centered_track_time


# Clip charge and combine into single data tensor for fitting.
fitting_event_data = jnp.array(event_data[['x', 'y', 'z', 'time', 'charge']].to_numpy())
print(fitting_event_data.shape)

# Setup likelihood
neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track)
print(neg_llh(track_src, centered_track_pos, centered_track_time, fitting_event_data))
negllh_time_v = jax.vmap(neg_llh, (None, None, 0, None), 0)

scale = 30.0
scale_rad = 30.0
@jax.jit
def neg_llh_5D(x, args):
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

solver = optx.BFGS(rtol=1e-8, atol=1e-4, use_inverse=True)
x0 = jnp.concatenate([track_src*scale_rad, centered_track_pos/scale])
best_x = optx.minimise(neg_llh_5D, solver, x0, throw=False).value
best_logl = neg_llh_5D(best_x, None)

print("best fit done. starting scan.")
print(best_logl)
x0 = centered_track_pos/scale

@jax.jit
def neg_llh_3D(x, args):
    track_dir = args[:3]
    track_time = args[3]
    return neg_llh(track_dir, x*scale, track_time, fitting_event_data)

def run_3D(track_dir, track_time):
    args = jnp.concatenate([track_dir, jnp.expand_dims(track_time, axis=0)])
    values = optx.minimise(neg_llh_3D, solver, x0, args=args, throw=False).value
    return neg_llh_3D(values, args), values * scale

run_3D_v = jax.jit(jax.vmap(run_3D, 0, (0, 0)))

zenith = jnp.linspace(track_src[0]-dzen, track_src[0]+dazi, n_eval)
azimuth = jnp.linspace(track_src[1]-dzen, track_src[1]+dazi, n_eval)
X, Y = jnp.meshgrid(zenith, azimuth)
init_dirs = jnp.column_stack([X.flatten(), Y.flatten()])

start = time.time()

# could find an optimal seed time for vertex
#dt = 500.
#tv = jnp.linspace(centered_track_time - dt, centered_track_time + 50, 100)
#
#def get_track_time(track_dir):
#    llh = negllh_time_v(track_dir, centered_track_pos, tv, fitting_event_data)
#    ix = jnp.argmin(llh, axis=0)
#    return ix
#
#get_track_time_v = jax.jit(jax.vmap(get_track_time, 0, 0))
#
#tic = time.time()
#n_splits = 25
#_init_dirs = init_dirs.reshape((n_splits, init_dirs.shape[0]//n_splits , init_dirs.shape[1]))
#indices = []
#for i, x in enumerate(_init_dirs):
#    print("done w timebatch: ", i)
#    indices.append(get_track_time_v(x))
#
#indices = jnp.concatenate(indices)
#init_times = jnp.take_along_axis(tv, indices, axis=0)
#toc = time.time()
#print(f"extraction of init times for grid took {toc-tic:.1f}s.")

init_times = jnp.ones_like(init_dirs[:, 0]) * centered_track_time

#tic = time.time()
#logls, sol_pos = run_3D_v(init_dirs, init_times)
#toc = time.time()
#print(f"jit + reco of grid took {toc-tic:.1f}s.")

if n_splits < 2:
    tic = time.time()
    logls, sol_pos = run_3D_v(init_dirs, init_times)
    logls.block_until_ready()
    toc = time.time()

else:
    tic0 = time.time()
    logls = []
    sol_poss = []
    n_per_split, r = divmod(len(init_dirs), n_splits)
    assert r==0, "The number of grid points need to be divisible by number of subgrids (splits)."

    for i in range(n_splits):
        tic = time.time()
        logls_, sol_pos_ = run_3D_v(init_dirs[i*n_per_split: (i+1) * n_per_split, :], init_times[i*n_per_split: (i+1) * n_per_split])
        logls_.block_until_ready()
        toc = time.time()
        logls.append(logls_)
        sol_poss.append(sol_pos_)
        print(f"jit + reco of sub-grid took {toc-tic:.1f}s.")

    logls = jnp.concatenate(logls, axis=0)
    sol_pos = jnp.concatenate(sol_poss, axis=0)
    tic = tic0

print(f"jit + reco of entire grid took {toc-tic:.1f}s.")

init_times = init_times.reshape(X.shape)
logls = logls.reshape(X.shape)
sol_x = sol_pos[:, 0].reshape(X.shape)
sol_y = sol_pos[:, 1].reshape(X.shape)
sol_z = sol_pos[:, 2].reshape(X.shape)

fig, ax = plt.subplots()
pc = ax.pcolormesh(np.rad2deg(X), np.rad2deg(Y), centered_track_time-init_times, vmin=-500, vmax=500, shading='auto', cmap=mpl.colormaps['seismic'])
cbar = fig.colorbar(pc)
cbar.ax.tick_params(labelsize=16)
cbar.ax.get_yaxis().labelpad = 5
cbar.set_label("true.t - init.t [ns]", fontsize=20)
cbar.outline.set_linewidth(1.5)

ax.set_xlabel("zenith [deg]", fontsize=16)
ax.set_ylabel("azimuth [deg]", fontsize=16)
ax.set_xlim(np.rad2deg([track_src[0]-dzen, track_src[0]+dzen]))
ax.set_ylim(np.rad2deg([track_src[1]-dazi, track_src[1]+dazi]))
ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=16)

ax.scatter(np.rad2deg(track_src[0]), np.rad2deg(track_src[1]), marker="*", color='red', label="truth", zorder=200)

plt.legend()
plt.tight_layout()
plt.savefig(f"mpe_scan_ev_{event_index}_w_noise_eqx_vertex_t.png", dpi=300)

fig, ax = plt.subplots()
pc = ax.pcolormesh(np.rad2deg(X), np.rad2deg(Y), centered_track_pos[0]-sol_x, vmin=-40, vmax=+40, shading='auto', cmap=mpl.colormaps['seismic'])
cbar = fig.colorbar(pc)
cbar.ax.tick_params(labelsize=16)
cbar.ax.get_yaxis().labelpad = 5
cbar.set_label("true.x - reco.x [m]", fontsize=20)
cbar.outline.set_linewidth(1.5)

ax.set_xlabel("zenith [deg]", fontsize=16)
ax.set_ylabel("azimuth [deg]", fontsize=16)
ax.set_xlim(np.rad2deg([track_src[0]-dzen, track_src[0]+dzen]))
ax.set_ylim(np.rad2deg([track_src[1]-dazi, track_src[1]+dazi]))
ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=16)

ax.scatter(np.rad2deg(track_src[0]), np.rad2deg(track_src[1]), marker="*", color='red', label="truth", zorder=200)

plt.legend()
plt.tight_layout()
plt.savefig(f"mpe_scan_ev_{event_index}_w_noise_eqx_vertex_x.png", dpi=300)

fig, ax = plt.subplots()
pc = ax.pcolormesh(np.rad2deg(X), np.rad2deg(Y), centered_track_pos[1]-sol_y, vmin=-40, vmax=+40, shading='auto', cmap=mpl.colormaps['seismic'])
cbar = fig.colorbar(pc)
cbar.ax.tick_params(labelsize=16)
cbar.ax.get_yaxis().labelpad = 5
cbar.set_label("true.y - reco.y [m]", fontsize=20)
cbar.outline.set_linewidth(1.5)

ax.set_xlabel("zenith [deg]", fontsize=16)
ax.set_ylabel("azimuth [deg]", fontsize=16)
ax.set_xlim(np.rad2deg([track_src[0]-dzen, track_src[0]+dzen]))
ax.set_ylim(np.rad2deg([track_src[1]-dazi, track_src[1]+dazi]))
ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=16)

ax.scatter(np.rad2deg(track_src[0]), np.rad2deg(track_src[1]), marker="*", color='red', label="truth", zorder=200)

plt.legend()
plt.tight_layout()
plt.savefig(f"mpe_scan_ev_{event_index}_w_noise_eqx_vertex_y.png", dpi=300)

fig, ax = plt.subplots()
pc = ax.pcolormesh(np.rad2deg(X), np.rad2deg(Y), centered_track_pos[2]-sol_z, vmin=-40, vmax=+40, shading='auto', cmap=mpl.colormaps['seismic'])
cbar = fig.colorbar(pc)
cbar.ax.tick_params(labelsize=16)
cbar.ax.get_yaxis().labelpad = 5
cbar.set_label("true.z - reco.z [m]", fontsize=20)
cbar.outline.set_linewidth(1.5)

ax.set_xlabel("zenith [deg]", fontsize=16)
ax.set_ylabel("azimuth [deg]", fontsize=16)
ax.set_xlim(np.rad2deg([track_src[0]-dzen, track_src[0]+dzen]))
ax.set_ylim(np.rad2deg([track_src[1]-dazi, track_src[1]+dazi]))
ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=16)

ax.scatter(np.rad2deg(track_src[0]), np.rad2deg(track_src[1]), marker="*", color='red', label="truth", zorder=200)

plt.legend()
plt.tight_layout()
plt.savefig(f"mpe_scan_ev_{event_index}_w_noise_eqx_vertex_z.png", dpi=300)

fig, ax = plt.subplots()
min_logl = np.amin(logls)
delta_logl = logls - np.amin(logls)
pc = ax.pcolormesh(np.rad2deg(X), np.rad2deg(Y), delta_logl, vmin=0, vmax=np.min([50, 1.2*np.amax(delta_logl)]), shading='auto', cmap=cx)
cbar = fig.colorbar(pc)
cbar.ax.tick_params(labelsize=16)
cbar.ax.get_yaxis().labelpad = 5
cbar.set_label("-2$\\Delta$log $L_{MPE}$", fontsize=20)
cbar.outline.set_linewidth(1.5)

ax.set_xlabel("zenith [deg]", fontsize=16)
ax.set_ylabel("azimuth [deg]", fontsize=16)
ax.set_xlim(np.rad2deg([track_src[0]-dzen, track_src[0]+dzen]))
ax.set_ylim(np.rad2deg([track_src[1]-dazi, track_src[1]+dazi]))
ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=16)

ax.scatter(np.rad2deg(track_src[0]), np.rad2deg(track_src[1]), marker="*", color='red', label="truth", zorder=200)

smpe_zenith = meta['spline_mpe_zenith']
smpe_azimuth = meta['spline_mpe_azimuth']
ax.scatter(np.rad2deg([smpe_zenith]), np.rad2deg([smpe_azimuth]), marker="x", color='lime', label='splineMPE')

zenith = best_x[0] / scale_rad
azimuth = best_x[1] / scale_rad
ax.scatter(np.rad2deg(zenith), np.rad2deg(azimuth), marker='+', color='magenta', label='bfgs')

contours = [4.61]
ix1, ix2 = np.where(delta_logl==0)
ax.scatter(np.rad2deg([X[ix1, ix2]]), np.rad2deg([Y[ix1, ix2]]), s=50, marker='o', facecolors='none', edgecolors='khaki', zorder=100., label='grid min')
ct = plt.contour(np.rad2deg(X), np.rad2deg(Y), delta_logl, levels=contours, linestyles=['solid'], colors=['khaki'], linewidths=1.0)

plt.legend()
plt.tight_layout()
plt.savefig(f"mpe_scan_ev_{event_id}_{event_index}_w_noise_small_eqx.png", dpi=300)


