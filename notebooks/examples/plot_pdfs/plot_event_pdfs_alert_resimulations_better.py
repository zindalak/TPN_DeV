import sys, os
sys.path.insert(0, "/home/storage/hans/jax_reco_gupta_corrections2")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from collections import defaultdict

import jax.numpy as jnp
from jax.scipy import optimize
import jax
jax.config.update("jax_enable_x64", True)
import optimistix as optx
import tensorflow_probability as tfp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lib.simdata_i3 import I3SimHandler
from lib.gupta_network_eqx import get_network_eval_v_fn
from lib.geo import cherenkov_cylinder_coordinates_w_rho_v
from lib.geo import get_xyz_from_zenith_azimuth
from lib.geo import impact_angle_cos_eta_v
from lib.gupta import c_multi_gupta_mpe_prob_midpoint2 as c_multi_gamma_mpe_prob_midpoint2
from lib.gupta import c_multi_gupta_spe_prob
from lib.plotting import adjust_plot_1d

from dom_track_eval import get_eval_network_doms_and_track

import time

from collections import defaultdict


dtype = jnp.float64
eval_network_v = get_network_eval_v_fn(bpath='/home/storage/hans/github/data/gupta/n96_w_penalty_1.e-3/new_model_no_penalties_tree_start_epoch_260.eqx', dtype=dtype, n_hidden=96)
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v, dtype=dtype, gupta=True)

event_ids = [12547,
    53129,
    29183,
    17424,
    10145,
    47271,
    39474,
    34495,
    49730,
    48963,
    59593,
    35147,
    54227,
    8789,
    51931,
    5352,
    49010,
    56061,
    37503
]

def make_event_plot(event_id):
    global eval_network_doms_and_track, geo
    bp = '/home/storage2/hans/i3files/lowE/ftr/'
    sim_handler = I3SimHandler(os.path.join(bp, f'meta_ds_event_{event_id}_from_0_to_10_1st_pulse.ftr'),
                              os.path.join(bp, f'pulses_ds_event_{event_id}_from_0_to_10_1st_pulse.ftr'),
                              '/home/storage/hans/jax_reco/data/icecube/detector_geometry.csv')

    pdf = PdfPages(f"pdfs_{event_id}.pdf")

    hit_sid = []
    hit_x = []
    hit_y = []
    hit_z = []
    hit_t = []
    hit_t_orig = []
    hit_q = []
    hit_all_t = []
    hit_all_q = []

    early_pulse_info = defaultdict(list)
    ct = 0

    for i in range(100):
        try:
            meta, pulses = sim_handler.get_event_data(i)
        except:
            continue

        #print(f"muon energy: {meta['muon_energy_at_detector']/1.e3:.1f} TeV")

        track_time = meta['muon_time']
        track_pos = [meta['muon_pos_x'], meta['muon_pos_y'], meta['muon_pos_z']]
        track_dir = [meta['muon_zenith'], meta['muon_azimuth']]

        # Get dom locations, first hit times, and total charges (for each dom).
        event_data_orig = sim_handler.get_per_dom_summary_from_sim_data(meta, pulses)
        event_data = sim_handler.get_per_dom_summary_from_sim_data(meta, pulses)
        # Remove early pulses.
        sim_handler.replace_early_pulse(event_data, pulses)
        #if not (event_data_orig['time'].to_numpy()==event_data['time'].to_numpy()).all():
        #    print ("removed an early pulse in event:", i)

        if i == 0:

            # Plot event.
            fig = plt.figure(figsize=(8,6))
            ax = plt.subplot(projection='3d')
            ax.set_xlabel('pos.x [m]', fontsize=16, labelpad=-25)
            ax.set_ylabel('pos.y [m]', fontsize=16, labelpad=-25)
            ax.set_zlabel('pos.z [m]', fontsize=16, labelpad=-25)
            ax.set_title(f'event {event_id}')

            df = event_data
            idx = df['charge'] > 0
            geo = sim_handler.geo

            try:
                im = ax.scatter(geo['x'], geo['y'], geo['z'], s=0.5, c='0.7', alpha=0.4)
            except:
                pass

            im = ax.scatter(df[idx]['x'], df[idx]['y'], df[idx]['z'], s=np.sqrt(df[idx]['charge']*100), c=df[idx]['time'],
                            cmap='rainbow_r',  edgecolors='k', zorder=1000)
            ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=16)
            cb = plt.colorbar(im, orientation="vertical", pad=0.1)
            cb.set_label(label='time [ns]', size='x-large')
            cb.ax.tick_params(labelsize='x-large')
            pdf.savefig(fig)

        hit_sid.append(event_data['sensor_id'].values)
        hit_t_orig.append(event_data_orig['time'].values - track_time)
        hit_t.append(event_data['time'].values - track_time)
        hit_x.append(event_data['x'].values)
        hit_y.append(event_data['y'].values)
        hit_z.append(event_data['z'].values)
        hit_q.append(event_data_orig['charge'].values)

        _tmp_hit_t = []
        _tmp_hit_q = []
        for s_id in event_data['sensor_id'].values:
            idx = pulses['sensor_id'] == s_id
            pulses_this_dom = pulses[idx]
            _tmp_hit_t.append(pulses_this_dom['time'].values - track_time)
            _tmp_hit_q.append(pulses_this_dom['charge'].values)

        hit_all_q.append(_tmp_hit_q)
        hit_all_t.append(_tmp_hit_t)

    dom_data = defaultdict(lambda: defaultdict(list))

    for i in range(len(hit_x)):
        for j in range(len(hit_x[i])):
            q = hit_q[i][j]

            x = hit_x[i][j]
            y = hit_y[i][j]
            z = hit_z[i][j]

            t = hit_t[i][j]
            t_orig = hit_t_orig[i][j]

            sid = hit_sid[i][j]
            dom_data[(x,y,z)]['first_hit_time'].append(t)
            dom_data[(x,y,z)]['first_hit_time_orig'].append(t_orig)
            dom_data[(x,y,z)]['q_tot'].append(q)
            dom_data[(x,y,z)]['sensor_id'].append(sid)
            for k, t in enumerate(hit_all_t[i][j]):
                dom_data[(x,y,z)]['pulse_time'].append(t)
                dom_data[(x,y,z)]['pulse_charge'].append(hit_all_q[i][j][k])

    for key in dom_data.keys():
        qs = dom_data[key]['q_tot']
        dom_data[key]['mean_q_tot'] = np.median(qs)
        dom_data[key]['sensor_id'] = dom_data[key]['sensor_id'][0] # replace list of duplicated sensor ids

    sorting = 'charge'

    dom_pos = []
    for key in dom_data.keys():
        dom_pos.append(jnp.array(key).reshape((1,3)))

    dom_pos = jnp.concatenate(dom_pos, axis=0)
    track_pos = jnp.array(track_pos)
    track_dir = jnp.array(track_dir)

    # evaluate network for these doms
    # notice that all true track vertex and directions are the same. So we use the first one.
    logits, av, bv, geo_time = eval_network_doms_and_track(dom_pos, track_pos, track_dir)
    mix_probs = jax.nn.softmax(logits)

    # get also the other DOM info
    track_dir_xyz = get_xyz_from_zenith_azimuth(track_dir)

    geo_time, closest_approach_dist, closest_approach_z, closest_approach_rho = \
                cherenkov_cylinder_coordinates_w_rho_v(dom_pos,
                                             track_pos,
                                             track_dir_xyz)

    cos_eta = impact_angle_cos_eta_v(dom_pos, track_pos, track_dir_xyz)

    # convert first_hit_times to delay_times by subtracting geo_times
    # and add more dom data to dom_data dict
    for i in range(len(dom_pos)):
        pos = tuple(np.array(dom_pos[i]))
        gt = geo_time[i]
        for j in range(len(dom_data[pos]['first_hit_time'])):
            dom_data[pos]['first_hit_time'][j] -= float(gt)
            dom_data[pos]['first_hit_time_orig'][j] -= float(gt)

        for j in range(len(dom_data[pos]['pulse_time'])):
            dom_data[pos]['pulse_time'][j] -= float(gt)

        dom_data[pos]['closest_approach_dist'] = closest_approach_dist[i]
        dom_data[pos]['closest_approach_rho'] = closest_approach_rho[i]
        dom_data[pos]['closest_approach_z'] = closest_approach_z[i]
        dom_data[pos]['cos_eta'] = cos_eta[i]
        dom_data[pos]['mix_probs'] = np.array(mix_probs[i])
        dom_data[pos]['a'] = np.array(av[i])
        dom_data[pos]['b'] = np.array(bv[i])

    # charge sorted mapping
    dom_positions = list(dom_data.keys())

    if sorting == 'charge':
        dom_positions.sort(key=lambda x: (len(dom_data[x]['q_tot']), np.mean(dom_data[x]['q_tot'])), reverse=True)
    else:
        dom_positions.sort(key=lambda x: dom_data[x]['closest_approach_dist'] ,reverse=False)

    n_plots = len(dom_positions)
    n_doms_per_page = 3
    xvals = np.linspace(-25, 3000, 30000)
    sigma = 3.0
    delta = 0.1

    def get_binned_charge(qtot):
        qmin, qmax = 0.99*np.min(qtot), 1.01*np.max(qtot)
        edges = np.linspace(qmin, qmax, 11)
        counts, _ = np.histogram(qtot, bins=edges)
        counts = counts / len(qtot)
        centers = 0.5 * (edges[1:] + edges[:-1])
        centers = jnp.clip(centers, min=1.0)
        return counts, centers

    c_multi_gamma_mpe_prob_midpoint2_vx = jax.jit(jax.vmap(c_multi_gamma_mpe_prob_midpoint2, (0, None, None, None, None, None), 0))
    c_multi_gamma_mpe_prob_midpoint2_vx_vn = jax.jit(jax.vmap(c_multi_gamma_mpe_prob_midpoint2_vx, (None, None, None, None, 0, None), 0))
    c_multi_gupta_spe_prob_vx = jax.jit(jax.vmap(c_multi_gupta_spe_prob, (0, None, None, None, None), 0))

    for i in range(0, min(100, len(dom_positions)), n_doms_per_page):
            print(i)
            fig, ax = plt.subplots(n_doms_per_page, 3)
            for j in range(n_doms_per_page):
                pos = tuple(dom_positions[i])
                g_mix_p = dom_data[pos]['mix_probs']
                g_a = dom_data[pos]['a']
                g_b = dom_data[pos]['b']
                mode = (g_a[1]-1)/g_b[1]

                plot_min = np.max([-50, np.min([np.min(dom_data[pos]['first_hit_time']), -25])])
                xvals = np.linspace(plot_min, np.max([1.5 * np.percentile(dom_data[pos]['first_hit_time'], [95])[0], 50]), 3000)

                dist = dom_data[pos]['closest_approach_dist']
                z = dom_data[pos]['closest_approach_z']
                rho = dom_data[pos]['closest_approach_rho']
                cos_eta = dom_data[pos]['cos_eta']

                sid = dom_data[pos]['sensor_id']

                n_p_orig = dom_data[pos]['mean_q_tot']
                n_p_orig = np.clip(n_p_orig, a_min = 1.0, a_max = None)


                for k in range(3):
                    tax = ax[j, k]
                    if k == 0:
                        #n_p_tmp = jnp.clip(n_p_orig, max = 3000.0)
                        #yval2 = c_multi_gamma_mpe_prob_midpoint2_vx(xvals, g_mix_p, g_a, g_b, n_p_tmp, sigma)

                        weights, centers = get_binned_charge(dom_data[pos]['q_tot'])
                        yval2 = c_multi_gamma_mpe_prob_midpoint2_vx_vn(xvals, g_mix_p, g_a, g_b, centers, sigma)
                        yval2 = jnp.sum(yval2 * np.expand_dims(weights, axis=1), axis=0)

                        norm2 = np.sum(yval2) * (xvals[1] - xvals[0])
                        tax.plot(xvals, yval2, label=f'MPE PDF (approx), q={n_p_orig}', color='black', linestyle='solid', lw=1)
                        tax.set_ylim([0.0, 1.2*np.amax([np.amax(yval2) / norm2])])

                        tax.set_xlabel('delay time [ns]')
                        tax.set_ylabel('pdf')

                        if k == 0:
                            time_key = 'first_hit_time'
                            tax.set_title(f"event {event_id} (dist={dist:.1f}m, z ={z:.0f}m, rho={np.rad2deg(rho):.0f}deg)", fontsize=6)

                        plot_min = np.max([-50, np.min([np.min(dom_data[pos][time_key]), -25])])
                        for tx in dom_data[pos][time_key]:
                            tax.axvline(tx, alpha=0.1, color='black', lw=0.5)

                        tax.hist(np.array(dom_data[pos][time_key]),
                                     bins = np.linspace(plot_min, xvals[-1], 21), density=True, alpha=0.5,
                                     label='first hit (from MC)', color='tab:green')

                        tax.legend(fontsize=4)
                        tax.set_xlim([plot_min, np.max([20, 1.5 * np.percentile(dom_data[pos]['first_hit_time'], [95])[0]])])

                    elif k == 1:
                        tmin = -25.
                        tmax = np.max(dom_data[pos]['pulse_time'])
                        tax.hist(dom_data[pos]['pulse_time'], bins=np.linspace(tmin, 2000, 200),
                                weights=dom_data[pos]['pulse_charge'], density = True, label='all pulses')

                        xvals = np.linspace(tmin, 1.5*tmax, 3000)
                        yval2 = c_multi_gupta_spe_prob_vx(xvals, g_mix_p, g_a, g_b, sigma)
                        tax.plot(xvals, yval2, label=f'SPE PDF', color='black', linestyle='solid', lw=1)
                        tax.set_xlabel('delay time [ns]')
                        tax.set_ylabel('pdf')


                        tax.set_xlim([tmin, 1.2*np.percentile(dom_data[pos]['pulse_time'], [95], weights=dom_data[pos]['pulse_charge'], method='inverted_cdf')[0]])
                        tax.set_ylim(ymin=1.e-4)
                        tax.set_yscale('log')
                        tax.set_title(f"impact angle cos_eta={cos_eta:.2f}", fontsize=6)
                        tax.legend(fontsize=6)

                    elif k == 2:
                        tax.hist(dom_data[pos]['q_tot'], color='tab:blue', histtype='step', lw=2)
                        tax.hist(dom_data[pos]['q_tot'], color='tab:blue', alpha=0.5)
                        tax.axvline(dom_data[pos]['mean_q_tot'], alpha=0.5, color='black', lw=1)
                        tax.set_xlabel('total charge [p.e.]')
                        tax.set_ylabel('counts')
                i+=1

            plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=1.0)

            pdf.savefig(fig)
            plt.close()

    pdf.close()
    return

for e_id in event_ids:
        make_event_plot(e_id)



