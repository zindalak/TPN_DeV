import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

import sys, os
sys.path.insert(0, "/home/storage/hans/jax_reco_new")
from lib.plotting import adjust_plot_1d
from lib.simdata_i3 import I3SimHandler
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages

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

outpath = "./results_pdf"

def get_cdf(logls, bins):
    logls.sort()
    n = len(logls)
    j = 0 # indexes into logl
    cdf_vals = np.zeros(len(bins)-1)
    count = 0
    for i in range(1, len(bins)):
        x = bins[i]
        while j < len(logls) and logls[j] <= x:
            count += 1
            j += 1

        cdf_vals[i-1] = float(count) / float(n)

    return cdf_vals

def get_hist(dat, bins):
    logls = dat[:, 0]
    return get_cdf(logls, bins), logls[-1]

def make_event_plot(event_id):
    pdf = PdfPages(os.path.join(outpath, f"reco_results_{event_id}.pdf"))
    starting_tracks = set([8, 1722, 9410, 19455, 21663, 25632, 27285, 32839, 40023, 51687, 57174, 59010, 59228, 63373])

    # Get data.
    bp = '/home/storage2/hans/i3files/alerts/bfrv2/nominal/'
    try:
        sim_handler = I3SimHandler(
            os.path.join(bp, f'meta_ds_event_{event_id}_N100_from_0_to_100_1st_pulse.ftr'),
            os.path.join(bp, f'pulses_ds_event_{event_id}_N100_from_0_to_100_1st_pulse.ftr'),
            '/home/storage/hans/jax_reco_new/data/icecube/detector_geometry.csv'
        )
    except:
        return

    splinempe_zen = []
    splinempe_azi = []
    for i in range(100):
        try:
            meta, pulses = sim_handler.get_event_data(i)
        except:
            break

        splinempe_zen.append(meta['spline_mpe_zenith'])
        splinempe_azi.append(meta['spline_mpe_azimuth'])

    try:
        event_data = sim_handler.get_per_dom_summary_from_index(0)
    except:
        return

    # Plot event.
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(projection='3d')
    ax.set_xlabel('pos.x [m]', fontsize=16, labelpad=-25)
    ax.set_ylabel('pos.y [m]', fontsize=16, labelpad=-25)
    ax.set_zlabel('pos.z [m]', fontsize=16, labelpad=-25)

    if int(event_id) in starting_tracks:
        ax.set_title(f'event {event_id} (starting track)')
    else:
        ax.set_title(f'event {event_id} (through-going track)')

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

    # Plot reco results.
    fig = plt.figure(figsize=(8, 6))
    title = ['original pulses', 'pre-pulses removed', 'pre-pulse removed + Qtot corrected']
    smpe_azi = splinempe_azi
    smpe_zen = splinempe_zen

    bp = '/home/storage/hans/jax_reco_new/examples/reco_realtime/update/delta_logl/large_scale/results/'
    dat_orig = np.load(os.path.join(bp, "original", f"llh_results_event_{event_id}_padded_input.npy"))
    dat_filter_prepulse = np.load(os.path.join(bp, "filter_prepulse", f"llh_results_event_{event_id}_padded_input.npy"))
    dat_filter_prepulse_corr = np.load(os.path.join(bp, "sigma_0.7", f"llh_results_event_{event_id}_padded_input.npy"))

    bins = np.linspace(0.0, 100.0, 400)
    bins_plot = bins + 0.5 * (bins[1] - bins[0])
    binc_plot = 0.5*(bins_plot[1:] + bins_plot[:-1])

    cdf_orig, m1 = get_hist(dat_orig, bins)
    cdf_filter_prepulse, m2 = get_hist(dat_filter_prepulse, bins)
    cdf_filter_prepulse_corr, m3 = get_hist(dat_filter_prepulse_corr, bins)

    plot_max = max([m1, m2, m3])+2
    xvals = np.linspace(0.0, plot_max, 1000)
    yvals = chi2.cdf(xvals, 2)

    ax = fig.add_subplot(2,2,1)
    ax.hist(binc_plot, bins=bins_plot, weights=cdf_orig, histtype='step', label='original pulses', lw=2)
    ax.hist(binc_plot, bins=bins_plot, weights=cdf_filter_prepulse, histtype='step', label='pre-pulses removed', lw=2)
    ax.hist(binc_plot, bins=bins_plot, weights=cdf_filter_prepulse_corr, histtype='step', label='pre-pulses removed + Qtot corrected (0.7)', lw=2)
    ax.plot(xvals, yvals, 'k-', label='$\\chi^2$ cdf', lw=1)
    ax.set_xlim([0.0, 22])
    ax.set_title(f'event {event_id}')
    ax.legend(loc='lower right', fontsize=6)
    ax.set_ylabel('CDF', fontsize=18)
    ax.set_xlabel('test-statistic', fontsize=18)
    ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=18)
    ax.yaxis.set_ticks_position('both')

    true = np.load(os.path.join(bp, "original", f"truecoords_event_{event_id}_padded_input.npy"))[0]
    dat_orig = np.load(os.path.join(bp, "original", f"mincoords_event_{event_id}_padded_input.npy"))
    dat_filter_prepulse = np.load(os.path.join(bp, "filter_prepulse", f"mincoords_event_{event_id}_padded_input.npy"))
    dat_filter_prepulse_corr = np.load(os.path.join(bp, "sigma_0.7", f"mincoords_event_{event_id}_padded_input.npy"))
    title = ['original pulses', 'pre-pulse filter', 'Qtot corr + pre-pulse filter']

    for i,f in enumerate([dat_orig, dat_filter_prepulse, dat_filter_prepulse_corr], 2):
        ax = fig.add_subplot(2,2,i)
        true_zen, true_azi = true[0], true[1]
        reco_zen, reco_azi = f[:, 0], f[:, 1]
        ax.scatter(np.rad2deg(true_azi), np.rad2deg(true_zen), marker='x', color='black', label='True', s=50, zorder=5)
        ax.scatter(np.rad2deg(smpe_azi), np.rad2deg(smpe_zen), label='spline MPE', s=7, color='tab:orange', alpha=0.3)
        ax.scatter(np.rad2deg(reco_azi), np.rad2deg(reco_zen), label='TPN MPE', s=7, color='tab:blue', alpha=0.3)
        ax.legend(fontsize=6)
        ax.set_xlabel('azimuth [deg]', fontsize=14)
        ax.set_ylabel('zenith [deg]', fontsize=14)
        ax.set_title(f"event {event_id} "+title[i-2])

        xmin = np.rad2deg(true_azi-min(np.min(reco_azi), np.min(smpe_azi)))
        xmax = np.rad2deg(max(np.max(reco_azi), np.max(smpe_azi))-true_azi)
        ymin = np.rad2deg(true_zen-min(np.min(reco_zen), np.min(smpe_zen)))
        ymax = np.rad2deg(max(np.max(reco_zen), np.max(smpe_zen))-true_zen)
        d_zen = max(ymin, ymax)+0.1
        d_azi = max(xmin, xmax)+0.1
        d_zen = min(d_zen, 5)
        d_azi = min(d_azi, 5)
        ax.set_xlim([np.rad2deg(true_azi)-d_azi, np.rad2deg(true_azi)+d_azi])
        ax.set_ylim([np.rad2deg(true_zen)-d_zen, np.rad2deg(true_zen)+d_zen])


    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=1.0)
    pdf.savefig(fig)

    pdf.close()

for event_id in event_ids:
    make_event_plot(event_id)

