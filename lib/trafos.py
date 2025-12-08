import jax
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
    return jnp.array([x0, x1, x2, x3, z, x, y])

transform_network_inputs_v = jax.jit(jax.vmap(transform_network_inputs, 0, 0))


def transform_network_outputs(x):
    a = 1.+20.*jax.nn.sigmoid(x[3:6]) + 1.e-20
    b = 2.0*jax.nn.sigmoid(x[6:9])
    logits = x[0:3]
    return logits, a, b

transform_network_outputs_v = jax.jit(jax.vmap(transform_network_outputs, 0, 0))


def transform_network_outputs_gupta(x):
    eps = 1.e-20
    a = 1.0 + jnp.exp(x[3:6]) + eps
    b = 1.0 / (1.e4*jax.nn.sigmoid(x[6:9]) + 0.1)
    logits = x[0:3]
    return logits, a, b

transform_network_outputs_gupta_v = jax.jit(jax.vmap(transform_network_outputs_gupta, 0, 0))


def transform_network_outputs_gupta_4comp(x):
    eps = 1.e-20
    # a = 1.0 + jnp.exp(x[4:8]) + eps
    a = 1.0 + jax.nn.softplus(x[4:8]) + eps
    b = 1.0 / (1.e4*jax.nn.sigmoid(x[8:12]) + 0.1)
    logits = x[0:4]
    return logits, a, b

transform_network_outputs_gupta_4comp_v = jax.jit(jax.vmap(transform_network_outputs_gupta_4comp, 0, 0))

"""
use transform_network_inputs and _outputs instead.


def transform_dimensions(dist, rho, z, zenith, azimuth, km_scale = 1000):
    # deprecated
    x0 = dist / km_scale
    x1 = jnp.cos(rho)
    x2 = jnp.sin(rho)
    x3 = z / km_scale

    z = jnp.cos(jnp.deg2rad(zenith))
    x = jnp.sin(jnp.deg2rad(zenith)) * jnp.cos(jnp.deg2rad(azimuth))
    y = jnp.sin(jnp.deg2rad(zenith)) * jnp.sin(jnp.deg2rad(azimuth))
    return jnp.array([x0, x1, x2, x3, z, x, y])


def transform_dimensions_vec(x, km_scale=1000):
    # deprecated
    # 0: dist, 1: rho, 2: z, 3: zenith, 4: azimuth
    dist = x[:, 0:1]
    rho = x[:, 1:2]
    z =  x[:, 2:3]
    zenith = x[:, 3:4]
    azimuth = x[:, 4:5]

    x0 = dist / km_scale
    x1 = jnp.cos(rho)
    x2 = jnp.sin(rho)
    x3 = z / km_scale

    x4 = jnp.cos(jnp.deg2rad(zenith))
    x5 = jnp.sin(jnp.deg2rad(zenith)) * jnp.cos(jnp.deg2rad(azimuth))
    x6 = jnp.sin(jnp.deg2rad(zenith)) * jnp.sin(jnp.deg2rad(azimuth))
    return jnp.concatenate([x0, x1, x2, x3, x4, x5, x6], axis=1)
"""
