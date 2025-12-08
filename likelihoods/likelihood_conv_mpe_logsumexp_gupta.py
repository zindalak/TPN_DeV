from lib.gupta import c_multi_gupta_mpe_logprob_midpoint2_stable_v
import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import pdf as norm_pdf
from jax.scipy.stats.norm import logpdf as norm_logpdf

def get_neg_c_triple_gamma_llh(eval_network_doms_and_track_fn, sigma=3.0):
    """
    here would be a smart docstring
    """

    sigma = jnp.array(sigma) # width of gaussian convolution

    @jax.jit
    def neg_c_triple_gamma_llh(track_direction,
                               track_vertex,
                               track_time,
                               event_data):


        # Constant parameters.
        sigma_noise = jnp.array(1000.0)

        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]
        #n_photons = jnp.round(charges + 0.5)
        #n_photons = jnp.clip(n_photons, min=1, max=1000)
        n_photons = jnp.clip(charges, min=1, max=10000)

        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)
        delay_time = first_hit_times - (geo_time + track_time)

        # Floor on negative time residuals.
        # Effectively a floor on the pdf.

        log_mix_probs = jax.nn.log_softmax(logits)
        log_physics_probs = c_multi_gupta_mpe_logprob_midpoint2_stable_v(delay_time,
                    log_mix_probs,
                    av,
                    bv,
                    n_photons,
                    sigma)


        log_floor_df = jnp.log(jnp.array(1./6000.))
        floor_weight = jnp.array(1.e-2)

        log_probs = jnp.concatenate([
                                        jnp.expand_dims(log_physics_probs, axis=0),
                                        jnp.expand_dims(jnp.ones_like(log_physics_probs) * log_floor_df, axis=0)
                                    ],
                                    axis=0
                                )

        weight = jnp.expand_dims(jnp.array([1.0-floor_weight, floor_weight]), axis=1)

        return -2.0 * jnp.sum(jax.scipy.special.logsumexp(log_probs, 0, weight))

    return neg_c_triple_gamma_llh



def get_llh_and_grad_fs_for_iminuit_migrad(eval_network_doms_and_track):
    """
    """
    neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track)

    @jax.jit
    def neg_llh_5D(x, track_time, data):
        track_direction = x[:2]
        track_vertex = x[2:]
        return neg_llh(track_direction, track_vertex, track_time, data)

    grad_neg_llh_5D = jax.jit(jax.grad(neg_llh_5D, argnums=0))

    return neg_llh_5D, grad_neg_llh_5D


def get_llh_and_grad_fs_for_iminuit_migrad_profile(eval_network_doms_and_track):
    """
    """
    neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track)

	# this gradient is 3D (vertex) not 5D (vertex + direction)
    grad_neg_llh_3D = jax.jit(jax.grad(neg_llh, argnums=1))

    return neg_llh, grad_neg_llh_3D
