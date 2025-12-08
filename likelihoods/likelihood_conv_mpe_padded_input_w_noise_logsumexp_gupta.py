from lib.gupta import c_multi_gupta_mpe_logprob_midpoint2_stable_v
from lib.gupta import c_multi_gupta_spe_prob_large_sigma_fine_v
from lib.gupta import c_multi_gupta_mpe_logprob_midpoint2_stable_large_sigma_v

import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import pdf as norm_pdf

def get_neg_c_triple_gamma_llh(eval_network_doms_and_track_fn, n_comp=3):
    """
    here would be a smart docstring
    """

    @jax.jit
    def neg_c_triple_gamma_llh(track_direction,
                               track_vertex,
                               track_time,
                               event_data):

        # Constant parameters.
        sigma = jnp.array(3.0) # width of gaussian convolution
        sigma_noise = jnp.array(1000.0)

        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]
        #n_photons = jnp.round(charges + 0.5)
        n_photons = jnp.clip(charges, min=1.0)

        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)
        delay_time = first_hit_times - (geo_time + track_time)

        # take care of padding
        idx_padded = event_data[:, 0] != 0.0
        idx_padded_s = idx_padded.reshape((idx_padded.shape[0], 1))

        # replace padded values with some computable outputs that don't lead to nan.
        logits = jnp.where(idx_padded_s, logits, jnp.ones((1, n_comp)))
        av = jnp.where(idx_padded_s, av, jnp.ones((1, n_comp))+3.0)
        bv = jnp.where(idx_padded_s, bv, jnp.ones((1, n_comp))*1.e-3)
        delay_time = jnp.where(idx_padded, delay_time, 10.0)

        log_mix_probs = jax.nn.log_softmax(logits)
        log_physics_probs = c_multi_gupta_mpe_logprob_midpoint2_stable_v(delay_time,
                    log_mix_probs,
                    av,
                    bv,
                    n_photons,
                    sigma)


        mix_probs = jnp.exp(log_mix_probs)
        log_noise_probs = jnp.log(c_multi_gupta_spe_prob_large_sigma_fine_v(delay_time,
                mix_probs,
                av,
                bv,
                sigma_noise))

        #log_noise_probs =  c_multi_gupta_mpe_logprob_midpoint2_stable_large_sigma_v(delay_time,
        #        log_mix_probs,
        #        av,
        #        bv,
        #        n_photons,
        #        sigma_noise)

        log_floor_df = jnp.log(jnp.array(1./6000.))
        floor_weight = jnp.array(0.001)
        noise_weight = jnp.array(0.01)

        log_probs = jnp.concatenate([
                                        jnp.expand_dims(log_physics_probs, axis=0),
                                        jnp.expand_dims(log_noise_probs, axis=0),
                                        jnp.expand_dims(jnp.ones_like(log_noise_probs) * log_floor_df, axis=0)
                                    ],
                                    axis=0
                                )

        weight = jnp.expand_dims(jnp.array([1.0-floor_weight-noise_weight, noise_weight, floor_weight]), axis=1)
        log_probs = jax.scipy.special.logsumexp(log_probs, 0, weight)
        log_probs = jnp.where(idx_padded, log_probs, jnp.array(0.0))
        return -2.0 * jnp.sum(log_probs)

    return neg_c_triple_gamma_llh
