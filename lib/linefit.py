import jax
import jax.numpy as jnp
import pandas as pd

def linefit_3d_time(fitting_event_data: pd.DataFrame):
    positions = fitting_event_data.values[:,2:5]
    times = fitting_event_data.values[:,5]
    """
    Perform a linefit in 3D space with time.
    Args:
        positions: Array of shape (N, 3) with (x, y, z) positions.
        times: Array of shape (N) with corresponding time values.
    Returns:
        r0: Fitted position at t0 (mean time).
        v: Velocity vector (direction and speed).
    """
    # Mean-center the data
    mean_pos = jnp.mean(positions, axis=0)
    mean_time = jnp.mean(times)
    # Subtract means
    delta_pos = positions - mean_pos
    delta_time = times - mean_time
    # Solve the least squares fit: v = (delta_pos^T delta_time) / (delta_time^T delta_time)
    numerator = jnp.dot(delta_pos.T, delta_time)  # shape (3,)
    denominator = jnp.dot(delta_time, delta_time) + 1e-8  # scalar, add epsilon for stability
    v = numerator / denominator
    t0 = mean_time
    r0 = mean_pos  # position at t0 = mean_time
    def direction_to_zenith_azimuth(dx, dy, dz):
        """Convert a direction vector to (zenith, azimuth) angles."""
        norm = jnp.sqrt(dx**2 + dy**2 + dz**2) + 1e-8  # avoid div-by-zero
        dx, dy, dz = dx / norm, dy / norm, dz / norm
        zenith = jnp.arccos(-dz)  # IceCube-style: zenith = 0 is downward
        azimuth = (jnp.arctan2(dy, dx) + jnp.pi) % (2 * jnp.pi)
        return jnp.array([zenith, azimuth])
    direction = direction_to_zenith_azimuth(*v)
    return r0, t0, v, direction