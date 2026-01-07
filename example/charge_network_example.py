
sys.path.insert(0, '/'.join(__file__.split('/')[:-2])+'/')

from lib.charge_network import get_charge_network_eval_v_fn

eval_charge_network_v = get_charge_network_eval_v_fn(bpath='/'.join(__file__.split('/')[:-2]) + '/data/absolute_charge_network', dtype=jnp.float32)

def eval_charge_network_2(dom_pos, track_vertex, track_direction):
    from lib.geo import get_xyz_from_zenith_azimuth
    track_dir_xyz = get_xyz_from_zenith_azimuth(track_direction)
    from lib.geo import cherenkov_cylinder_coordinates_w_rho2_v
    geo_time, closest_approach_dist, closest_approach_z, closest_approach_rho = \
            cherenkov_cylinder_coordinates_w_rho2_v(dom_pos,
                                         track_vertex,
                                         track_dir_xyz)
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
    return y
