import sys, os
path = '/'.join(__file__.split('/')[:-2])

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
modelpath = '/home/mjansson/kod/TPN_DeV/data/gupta/new_model_no_penalties_tree_start_epoch_100.eqx'

import jax.numpy as jnp
dtype = jnp.float64
n_hidden = 96
gupta = True
n_comp = 4
GAUS_CONV_WIDTH = 3

from lib.gupta_network_eqx_4comp import get_network_eval_v_fn, get_network_eval_v_fn_f32
eval_network_v = get_network_eval_v_fn_f32(bpath=modelpath, dtype=dtype, n_hidden=n_hidden)
from dom_track_eval import get_eval_network_doms_and_track
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v, dtype=dtype, gupta=gupta, n_comp=n_comp)
from likelihoods.likelihood_conv_mpe_logsumexp_gupta import get_neg_c_triple_gamma_llh
neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track, sigma=GAUS_CONV_WIDTH)

import pandas as pd
geo_file = '/'.join(__file__.split('/')[:-2])+'/data/icecube/detector_geometry.csv'
geo = pd.read_csv(geo_file)

from fitting.llh_fitter import get_fitter
fit_llh = get_fitter(
                    neg_llh,
                    use_multiple_vertex_seeds = True,
                    prescan_time = True
                )

import jax
fit_llh_jit = jax.jit(fit_llh)


def reconstructEvent(event_data, track_pos, track_time, track_src):
    import numpy as np
    from lib.geo import center_track_pos_and_time_based_on_data
    centered_track_pos, centered_track_time = center_track_pos_and_time_based_on_data(event_data, track_pos, track_time, track_src)
    fitting_event_data = jnp.array(event_data[['x', 'y', 'z', 'time', 'charge']].to_numpy())
    solution = fit_llh_jit(track_src, centered_track_pos, centered_track_time, fitting_event_data)
    best_logl, best_direction, best_vertex, best_time = solution
    return best_logl, best_direction, best_vertex, best_time

def example():
    from jax import random
    import jax.numpy as jnp
    from example.generate_event import generateEvent
    seed = 0
    key = random.key(seed)
    track_vertex = jnp.array([0,0,0])
    track_src = jnp.array([1,1])
    energy_scale = 2
    event, key = generateEvent(key, track_vertex, track_src, energy_scale)
    track_time = 0
    import pandas as pd
    df = pd.DataFrame(event, columns=['x', 'y', 'z', 'time', 'charge'])
    # from example.reconstruct_event import reconstructEvent
    best_logl, best_direction, best_vertex, best_time = reconstructEvent(df, track_vertex, track_time, track_src)
    return best_logl, best_direction, best_vertex, best_time
    