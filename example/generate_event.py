path = '/'.join(__file__.split('/')[:-2])

def getDOMLocations(filepath):
    import pandas as pd
    geo_file = path +'/data/icecube/detector_geometry.csv'
    geo = pd.read_csv(geo_file)
    return geo

def getExpectedCharge(track_vertex, track_direction, dom_pos):
    global y
    global x_prime
    global x_bis
    import jax.numpy as jnp
    def transform_network_inputs(x):
        # 0: dist, 1: rho, 2: z, 3: zenith, 4: azimuth
        # in units of m and radians
        km_scale = 1000
        dist = x[0]
        rho = x[1]
        z = x[2]
        zenith = x[3]
        azimuth = x[4]
        x0 = dist / km_scale
        x1 = jnp.cos(rho)
        x2 = jnp.sin(rho)
        x3 = z / km_scale
        z = jnp.cos(zenith)
        x = jnp.sin(zenith) * jnp.cos(azimuth)
        y = jnp.sin(zenith) * jnp.sin(azimuth)
        return jnp.array([x, y, z, x0, x1, x2, x3])
    from lib.charge_network import get_charge_network_eval_v_fn
    eval_charge_network_v = get_charge_network_eval_v_fn(bpath= path + '/data/absolute_charge_network', dtype=jnp.float32)
    from lib.geo import get_xyz_from_zenith_azimuth
    import jax
    track_dir_xyz = get_xyz_from_zenith_azimuth(track_direction)
    from lib.geo import cherenkov_cylinder_coordinates_w_rho2_v
    geo_time, closest_approach_dist, closest_approach_z, closest_approach_rho = \
        cherenkov_cylinder_coordinates_w_rho2_v(dom_pos,
            track_vertex,
            track_dir_xyz)
    transform_network_inputs_v = jax.jit(jax.vmap(transform_network_inputs, (0, ), 0))
    track_zenith = track_direction[0]
    track_azimuth = track_direction[1]
    x_prime = jnp.column_stack([closest_approach_dist,
        closest_approach_rho,
        closest_approach_z,
        jnp.repeat(track_zenith, len(closest_approach_dist)),
        jnp.repeat(track_azimuth, len(closest_approach_dist))])
    x_bis = transform_network_inputs_v(x_prime)
    y = eval_charge_network_v(x_bis)
    y = jax.lax.select(x_prime[:,0].reshape((5160,1))<400, y, jnp.zeros(y.shape))
    return y

def getPoisson(mean_charges, key):
    from jax import random
    charges = random.poisson(key, mean_charges)
    return charges

def cdf(mix_probs, a, b, x):
    from lib.gupta import cdf
    cdf_value = cdf(x, a[0], b[0])*mix_probs[0]
    cdf_value += cdf(x, a[1], b[1])*mix_probs[1]
    cdf_value += cdf(x, a[2], b[2])*mix_probs[2]
    cdf_value += cdf(x, a[3], b[3])*mix_probs[3]
    return cdf_value

def binarysearch(logits, a, b, delay_cdf, depth):
    import jax
    mix_probs = jax.nn.softmax(logits)
    minvalue = 0.0
    midpoint = 400.0
    maxvalue = 6000.0
    step = 0
    while step < depth:
        cdf_value = cdf(mix_probs, a, b, midpoint)
        minvalue = jax.lax.select(delay_cdf < cdf_value, minvalue, midpoint)
        maxvalue = jax.lax.select(delay_cdf > cdf_value, maxvalue, midpoint)
        midpoint = (maxvalue + minvalue)/2
        step += 1
    return midpoint

import jax
binarysearch_v = jax.vmap(binarysearch, (0, 0, 0, 0, None), 0)

def getUnfoldedCDF(photons, key, shape):
    from jax import random
    random_values = random.uniform(key, shape)
    import jax.numpy as jnp
    unfolded_cdf = 1 - jnp.power(1-random_values, 1.0/photons)
    return unfolded_cdf

def getdelaytimes(photons, dom_pos, track_vertex, track_direction, key1, key2):
    import jax.numpy as jnp
    modelpath = path + '/data/gupta/new_model_no_penalties_tree_start_epoch_100.eqx'
    from lib.gupta_network_eqx_4comp import get_network_eval_v_fn_f32
    eval_network_v = get_network_eval_v_fn_f32(bpath=modelpath, dtype=jnp.float64, n_hidden=96)
    from lib.geo import get_xyz_from_zenith_azimuth
    track_dir_xyz = get_xyz_from_zenith_azimuth(track_direction)
    from lib.geo import cherenkov_cylinder_coordinates_w_rho2_v
    geo_time, closest_approach_dist, closest_approach_z, closest_approach_rho = \
        cherenkov_cylinder_coordinates_w_rho2_v(dom_pos,
            track_vertex,
            track_dir_xyz)
    from lib.trafos import transform_network_inputs_v
    track_zenith = track_direction[0]
    track_azimuth = track_direction[1]
    x = jnp.column_stack([closest_approach_dist,
                      closest_approach_rho,
                      closest_approach_z,
                      jnp.repeat(track_zenith, len(closest_approach_dist)),
                      jnp.repeat(track_azimuth, len(closest_approach_dist))])
    x = jnp.array(x, dtype=jnp.float64)
    x_prime = transform_network_inputs_v(x)
    y_pred = eval_network_v(x_prime)
    from lib.trafos import transform_network_outputs_gupta_4comp_v as transform_network_outputs_v
    logits, av, bv = transform_network_outputs_v(y_pred)
    unfolded_cdf = getUnfoldedCDF(photons, key1, closest_approach_dist.shape)
    from jax import random
    delaytimes = binarysearch_v(logits, av, bv, unfolded_cdf, 9)
    delaytimes += random.normal(key2, delaytimes.shape)*3
    delaytimes += geo_time
    #jnp.array(event_data[['x', 'y', 'z', 'time', 'charge']].to_numpy())
    #return photons, random_values, y_pred
    return delaytimes, unfolded_cdf, logits, av, bv



'''
get expected charges
get poison distvalues
get values between 0 and 1
get analytical CDF of extreme value dist for uniform
do a fix step binary search in cdf
'''



def generateEvent(key, track_vertex, track_direction, energy_scale):
    import jax.numpy as jnp
    #path =  '/home/mjansson/kod/TPN_DeV'
    DOMs = getDOMLocations(path)
    DOMs = jnp.array(DOMs[['x', 'y', 'z']].to_numpy())
    mean_charges = getExpectedCharge(track_vertex, track_direction, DOMs)
    from jax import random
    key, subkey = random.split(key)
    actual_charges = getPoisson(mean_charges*energy_scale, subkey)
    charges = actual_charges[jnp.nonzero(actual_charges)]
    hit_doms = DOMs[jnp.nonzero(actual_charges)[0],:]
    key, subkey1 = random.split(key)
    key, subkey2 = random.split(key)
    delaytimes, unfolded_cdf, logits, av, bv = getdelaytimes(charges, hit_doms, track_vertex, track_direction, subkey1, subkey2)
    jnp.mean(unfolded_cdf)
    #jnp.array(event_data[['x', 'y', 'z', 'time', 'charge']].to_numpy())
    result = jnp.array([hit_doms[:,0], hit_doms[:,1], hit_doms[:,2], delaytimes, charges]).transpose()
    return result, key

def example():
    from jax import random
    import jax.numpy as jnp
    seed = 0
    key = random.key(seed)
    track_vertex = jnp.array([0,0,0])
    track_direction = jnp.array([1,1])
    energy_scale = 2
    event1, key = generateEvent(key, track_vertex, track_direction, energy_scale)
    event2, key = generateEvent(key, track_vertex, track_direction, energy_scale)
    event3, key = generateEvent(key, track_vertex, track_direction, energy_scale)
