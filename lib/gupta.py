import jax.numpy as jnp
import jax

from jax.scipy.stats.norm import pdf as norm_pdf
from jax.scipy.stats.norm import logpdf as norm_logpdf
from jax.scipy.special import logsumexp

from quadax import quadgk as jquad

def log1m_exp(x):
    """
    Numerically stable calculation
    of the quantity log(1 - exp(x)),
    following the algorithm of
    Machler [1]. This is
    the algorithm used in TensorFlow Probability,
    PyMC, and Stan, but it is not provided
    yet with Numpyro.

    Currently returns NaN for x > 0,
    but may be modified in the future
    to throw a ValueError

    [1] https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    # return 0. rather than -0. if
    # we get a negative exponent that exceeds
    # the floating point representation
    crit = -0.6931472
    arr_x = 1.0 * jnp.array(x)

    crit_oob = jnp.log(jnp.finfo(
        arr_x.dtype).smallest_normal)+5
    oob = arr_x < crit_oob
    mask = arr_x > crit

    more_val = jnp.log(-jnp.expm1(jnp.clip(arr_x, a_min=crit)))
    less_val = jnp.log1p(-jnp.exp(jnp.clip(arr_x, a_max=crit)))

    return jnp.where(
        oob,
        -jnp.exp(crit_oob),
        jnp.where(
            mask,
            more_val,
            less_val))

def log_cdf(x, a, b):
    return a * log1m_exp(-b*x)

def log_sf(x, a, b):
    return log1m_exp(log_cdf(x, a, b))

def log_pdf(x, a, b):
    return jnp.log(a) + jnp.log(b) + (a-1.) * log1m_exp(-b*x) - b*x

def cdf(x, a, b):
    return jnp.power(1.-jnp.exp(-b*x), a)

def sf(x, a, b):
    return 1. - cdf(x, a, b)

def pdf(x, a, b):
    return a*b*jnp.power(1.-jnp.exp(-b*x), a-1) * jnp.exp(-b*x)

def multi_gupta_pdf(x, mix_probs, a, b):
    x = jnp.expand_dims(x, axis=0)
    a = jnp.expand_dims(a, axis=-1)
    b = jnp.expand_dims(b, axis=-1)
    mix_probs = jnp.expand_dims(mix_probs, axis=-1)
    pdf_vals = pdf(x, a, b)
    return jnp.squeeze(jnp.sum(mix_probs * pdf_vals, axis=0))

def multi_gupta_cdf(x, mix_probs, a, b):
    x = jnp.expand_dims(x, axis=0)
    a = jnp.expand_dims(a, axis=-1)
    b = jnp.expand_dims(b, axis=-1)
    mix_probs = jnp.expand_dims(mix_probs, axis=-1)
    cdf_vals = cdf(x, a, b)
    return jnp.squeeze(jnp.sum(mix_probs * cdf_vals, axis=0))

# added for SRT
def log_weighted_avg_two(logA, logB, wA, wB, eps=1e-30):
    """ (wA*A + wB*B)/(wA+wB) in log-space. """
    wA = jnp.maximum(wA, 0.0)
    wB = jnp.maximum(wB, 0.0)
    denom = jnp.maximum(wA + wB, eps)
    return jax.scipy.special.logsumexp(
        jnp.stack([logA + jnp.log(jnp.maximum(wA, eps)),
                   logB + jnp.log(jnp.maximum(wB, eps))], axis=0),
        axis=0
    ) - jnp.log(denom)

# added for SRT
def log_weighted_avg_three(logA, logB, logC, wA, wB, wC, eps=1e-30):
    """ (wA*A + wB*B + wC*C)/(wA+wB+wC) in log-space. """
    wA = jnp.maximum(wA, 0.0)
    wB = jnp.maximum(wB, 0.0)
    wC = jnp.maximum(wC, 0.0)
    denom = jnp.maximum(wA + wB + wC, eps)
    return jax.scipy.special.logsumexp(
        jnp.stack([logA + jnp.log(jnp.maximum(wA, eps)),
                   logB + jnp.log(jnp.maximum(wB, eps)),
                   logC + jnp.log(jnp.maximum(wC, eps))], axis=0),
        axis=0
    ) - jnp.log(denom)

# added for SRT
def srt_get_noise_weights(expected_pes, eps=1e-12):
    log10_pes = jnp.log10(jnp.maximum(expected_pes, eps))

    physicsW = 1.0 - jnp.exp(
        -jnp.power(
            10.0,
            0.774234
            - 1.02385 * jnp.arctan(0.969486 - 0.577865 * log10_pes)
            + 0.193763 * jnp.arctan(16.2363 + 4.76944 * log10_pes),
        )
    )

    rw_arg = -1.36638 - 1.10099 * jnp.arctan(1.08653 + 0.850798 * log10_pes)
    rw_arg = jnp.minimum(0.0, rw_arg)
    randomW = 1.0 - jnp.exp(-jnp.power(10.0, rw_arg))

    afterW = 1.0 - jnp.exp(
        -jnp.power(
            10.0,
            -0.186922
            + 0.946511 * log10_pes
            - 1.08804 * jnp.arctan(1.73241 + 0.760878 * log10_pes),
        )
    )

    preLateW = 1.0 - jnp.exp(
        -jnp.power(
            10.0,
            0.144828
            + 0.928378 * log10_pes
            - 1.11346 * jnp.arctan(1.3868 + 0.780568 * log10_pes),
        )
    )
    return physicsW, randomW, afterW, preLateW

# added for SRT
def build_weighted_logdf_srt_mpe(
    *,
    log_bareDF,
    log_stochDF,
    log_rDF,
    log_afterDF,
    log_floorDF,
    bare_pes,
    stoch_pes,
    expected_pes,
    modelStochastics: bool,
    noiseModel: str,      # "none" | "flat" | "SRT" 등
    floorWeight: float,
    # low_pes 포함하고 싶으면 enEstimate를 넣어 확장 가능
    low_pes: float = 0.0,
    eps=1e-30,
):
    # 1) physicsDF
    w_bare = bare_pes
    w_stoch = stoch_pes + low_pes

    log_physicsDF = jnp.where(
        modelStochastics,
        log_weighted_avg_two(log_stochDF, log_bareDF, w_stoch, w_bare, eps=eps),
        log_bareDF
    )

    # 2) randomDF (noiseModel != "none"일 때만 의미 있음)
    bumpWeight = jnp.where(noiseModel == "flat", 0.0, 1.0)
    bump_amp = (1.0 - jnp.exp(-expected_pes))                 # scalar/array
    w_r = bumpWeight * bump_amp
    w_floor = floorWeight

    log_randomDF = log_weighted_avg_two(log_rDF, log_floorDF, w_r, w_floor, eps=eps)

    # 3) SRT weights
    physicsW, randomW, afterW, preLateW = srt_get_noise_weights(expected_pes)

    # physicsW + preLateW 를 physicsDF에 곱하는 효과
    w_phys = physicsW + preLateW
    w_rand = randomW
    w_after = afterW

    # noiseModel == "none"이면 사실상 physics만 써야 함
    def _no_noise(_):
        return log_physicsDF

    def _with_noise(_):
        return log_weighted_avg_three(
            log_physicsDF, log_randomDF, log_afterDF,
            w_phys, w_rand, w_after,
            eps=eps
        )

    return jax.lax.cond(noiseModel == "none", _no_noise, _with_noise, operand=None)


def c_multi_gupta_mpe_logprob_midpoint2_stable(x, log_mix_probs, a, b, n, sigma=3.0):
    """
    Q < 30
    """
    nmax = 10
    nint1 = 10
    nint2 = 15
    nint3 = 35
    #eps = 1.e-12
    eps = 1.e-6

    int_scale = 1

    x0 = eps
    x_m0 = 0.01
    xvals0 = jnp.linspace(x0, x_m0, 10 * int_scale)[:-1]

    x_m1 = 0.05
    xvals1 = jnp.linspace(x_m0, x_m1, 10 * int_scale)[:-1]

    x_m2 = 0.25
    xvals2 = jnp.linspace(x_m1, x_m2, 10 * int_scale)[:-1]

    x_m25 = 0.75
    xvals25 = jnp.linspace(x_m2, x_m25, 10 * int_scale)[:-1]

    x_m3 = 2.5
    xvals3 = jnp.linspace(x_m25, x_m3, 10 * int_scale)[:-1]

    x_m4 = 8.0
    xvals4 = jnp.linspace(x_m3, x_m4, 20 * int_scale)

    xmin = jnp.max(jnp.array([1.5 * eps, x - 10 * sigma * int_scale]))
    xmax = jnp.max(jnp.array([xmin+1.5*eps, x + 10 * sigma * int_scale]))
    xvals_x = jnp.linspace(xmin, xmax, 101 * int_scale)

    xvals = jnp.sort(jnp.concatenate([xvals0, xvals1, xvals2, xvals25, xvals3, xvals4, xvals_x]))

    dx = xvals[1:]-xvals[:-1]

    xvals = 0.5*(xvals[:-1]+xvals[1:])
    log_n_pdf = norm_logpdf(xvals, loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    log_mix_probs_e = jnp.expand_dims(log_mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    log_pdfs = logsumexp(log_pdf(xvals_e, a_e, b_e) + log_mix_probs_e, 0)
    log_sfs = logsumexp(log_sf(xvals_e, a_e, b_e) + log_mix_probs_e, 0)

    return logsumexp(log_n_pdf + log_pdfs + (n-1) * log_sfs + jnp.log(dx) + jnp.log(n), 0)

c_multi_gupta_mpe_logprob_midpoint2_stable_v = jax.vmap(c_multi_gupta_mpe_logprob_midpoint2_stable, (0, 0, 0, 0, 0, None), 0)

def c_multi_gupta_mpe_prob_midpoint2(x, mix_probs, a, b, n, sigma=3.0):
    """
    Q < 30
    """
    nmax = 10
    nint1 = 10
    nint2 = 15
    nint3 = 35
    #eps = 1.e-12
    eps = 1.e-6

    x0 = eps
    x_m0 = 0.01
    xvals0 = jnp.linspace(x0, x_m0, 10)

    x_m1 = 0.05
    xvals1 = jnp.linspace(x_m0, x_m1, 10)

    x_m2 = 0.25
    xvals2 = jnp.linspace(x_m1, x_m2, 10)

    x_m25 = 0.75
    xvals25 = jnp.linspace(x_m2, x_m25, 10)

    x_m3 = 2.5
    xvals3 = jnp.linspace(x_m25, x_m3, 10)

    x_m4 = 8.0
    xvals4 = jnp.linspace(x_m3, x_m4, 20)

    #x_m5 = 6000.0
    #xvals5 = jnp.linspace(x_m4, x_m5, 20)

    #xmin = jnp.max(jnp.array([1.5 * eps, x - 4 * sigma]))
    #xmax = jnp.max(jnp.array([xmin+1.5*eps, x + 4 * sigma]))
    #xvals_x = jnp.linspace(xmin, xmax, 30)
    #xvals = jnp.sort(jnp.concatenate([xvals0, xvals1, xvals2, xvals25, xvals3, xvals4, xvals5, xvals_x]))
    xmin = jnp.max(jnp.array([1.5 * eps, x - 10 * sigma]))
    xmax = jnp.max(jnp.array([xmin+1.5*eps, x + 10 * sigma]))
    xvals_x = jnp.linspace(xmin, xmax, 101)
    xvals = jnp.sort(jnp.concatenate([xvals0, xvals1, xvals2, xvals25, xvals3, xvals4, xvals_x]))

    dx = xvals[1:]-xvals[:-1]

    xvals = 0.5*(xvals[:-1]+xvals[1:])
    n_pdf = norm_pdf(xvals, loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    sfs = jnp.sum(mix_probs_e * sf(xvals_e, a_e, b_e), axis=0)
    pdfs = jnp.sum(mix_probs_e * jnp.clip(pdf(xvals_e, a_e, b_e), min=0, max=None), axis=0)

    return jnp.sum(n_pdf * n * pdfs * jnp.power(sfs, n-1.0) * dx)

c_multi_gupta_mpe_prob_midpoint2_v = jax.vmap(c_multi_gupta_mpe_prob_midpoint2, (0, 0, 0, 0, 0, None), 0)


def c_multi_gupta_spe_prob(x, mix_probs, a, b, sigma=3.0):
    nmax = 10
    nint1 = 20
    nint2 = 30
    nint3 = 70
    eps = 1.e-6

    xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
    diff = xmax-x
    xmin = jnp.max(jnp.array([jnp.array(0.0), x - diff]))
    x_m1 = xmin + 0.02*sigma
    x_m2 = x_m1 + 0.5*sigma

    # two combined the two integration regions
    xvals = jnp.concatenate([jnp.linspace(xmin, x_m1, nint1),
                             jnp.linspace(x_m1, x_m2, nint2),
                             jnp.linspace(x_m2, xmax, nint3)])

    dx = xvals[1:]-xvals[:-1]

    xvals = 0.5*(xvals[:-1]+xvals[1:])
    n_pdf = norm_pdf(xvals, loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    pdfs = jnp.sum(mix_probs_e * jnp.clip(pdf(xvals_e, a_e, b_e), min=0, max=None), axis=0)

    return jnp.sum(n_pdf * pdfs * dx)

c_multi_gupta_spe_prob_v = jax.vmap(c_multi_gupta_spe_prob, (0, 0, 0, 0, None), 0)


def c_multi_gupta_spe_prob_large_sigma_fine(x, mix_probs, a, b, sigma=1000.):
    """
    ... for noise. tested for sigma of order 1000.
    """
    nmax = 6
    nint1 = 20
    nint2 = 30
    nint3 = 70
    eps = 1.e-6

    xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
    diff = xmax-x
    xmin = jnp.max(jnp.array([jnp.array(0.0), x - diff]))
    x_m1 = xmin + 10
    x_m2 = x_m1 + 100

    # two combined the two integration regions
    xvals = jnp.concatenate([jnp.linspace(xmin, x_m1, nint1),
                             jnp.linspace(x_m1, x_m2, nint2),
                             jnp.linspace(x_m2, xmax, nint3)])

    dx = xvals[1:]-xvals[:-1]

    xvals = 0.5*(xvals[:-1]+xvals[1:])
    n_pdf = norm_pdf(xvals, loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    pdfs = jnp.sum(mix_probs_e * jnp.clip(pdf(xvals_e, a_e, b_e), min=0, max=None), axis=0)

    return jnp.sum(n_pdf * pdfs * dx)

c_multi_gupta_spe_prob_large_sigma_fine_v = jax.vmap(c_multi_gupta_spe_prob_large_sigma_fine, (0, 0, 0, 0, None), 0)

def multi_gupta_mpe_pdf(x, mix_probs, a, b, n):
    pdf_vals = multi_gupta_pdf(x, mix_probs, a, b)
    cdf_vals = multi_gupta_cdf(x, mix_probs, a, b)
    return n * pdf_vals * jnp.power(1.-cdf_vals, n-1)

def _integrand(x, mix_probs, a, b, n, sigma, t):
    return norm_pdf(x, loc=t, scale=sigma) * multi_gupta_mpe_pdf(x, mix_probs, a, b, n)

def c_multi_gupta_mpe_prob_quad(x, mix_probs, a, b, n, sigma):
    # define integration range in units of sigma
    delta = jnp.array(15) # units of sigma
    # and stay away from x = 0
    eps = jnp.array(1.e-12)

    xmax = jnp.max(jnp.array([delta*sigma, x + delta*sigma]))
    diff = xmax - x
    xmin = jnp.max(jnp.array([jnp.array(0.0)+eps, x - diff]))

    res = jquad(_integrand,
                 jnp.array([xmin, 0.1, 1.0, xmax]),
                 args=(mix_probs, a, b, n, sigma, x),
                 epsabs=1.e-4,
                 epsrel=1.e-4,
                 order=51,
                 max_ninter=5
              )[0]

    return res

c_multi_gupta_mpe_prob_quad_v = jax.vmap(c_multi_gupta_mpe_prob_quad, (0, 0, 0, 0, 0, None), 0)

def c_multi_gupta_mpe_logprob_midpoint2_stable_large_sigma(x, log_mix_probs, a, b, n, sigma=3.0):
    """
    Q < 30
    """
    nmax = 3
    eps = 1.e-6

    int_scale = 1

    x0 = eps
    x_m0 = 0.01
    xvals0 = jnp.linspace(x0, x_m0, 10 * int_scale)[:-1]

    x_m1 = 0.05
    xvals1 = jnp.linspace(x_m0, x_m1, 10 * int_scale)[:-1]

    x_m2 = 0.25
    xvals2 = jnp.linspace(x_m1, x_m2, 10 * int_scale)[:-1]

    x_m25 = 0.75
    xvals25 = jnp.linspace(x_m2, x_m25, 10 * int_scale)[:-1]

    x_m3 = 2.5
    xvals3 = jnp.linspace(x_m25, x_m3, 10 * int_scale)[:-1]

    x_m4 = 8.0
    xvals4 = jnp.linspace(x_m3, x_m4, 20 * int_scale)

    xmin = jnp.max(jnp.array([1.5 * eps, x - nmax * sigma * int_scale]))
    xmax = jnp.max(jnp.array([xmin+1.5*eps, x + nmax * sigma * int_scale]))
    xvals_x = jnp.linspace(xmin, xmax, 101 * int_scale)

    xvals = jnp.sort(jnp.concatenate([xvals0, xvals1, xvals2, xvals25, xvals3, xvals4, xvals_x]))

    dx = xvals[1:]-xvals[:-1]

    xvals = 0.5*(xvals[:-1]+xvals[1:])
    log_n_pdf = norm_logpdf(xvals, loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    log_mix_probs_e = jnp.expand_dims(log_mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    log_pdfs = logsumexp(log_pdf(xvals_e, a_e, b_e) + log_mix_probs_e, 0)
    log_sfs = logsumexp(log_sf(xvals_e, a_e, b_e) + log_mix_probs_e, 0)

    return logsumexp(log_n_pdf + log_pdfs + (n-1) * log_sfs + jnp.log(dx) + jnp.log(n), 0)

c_multi_gupta_mpe_logprob_midpoint2_stable_large_sigma_v = jax.vmap(c_multi_gupta_mpe_logprob_midpoint2_stable_large_sigma, (0, 0, 0, 0, 0, None), 0)
