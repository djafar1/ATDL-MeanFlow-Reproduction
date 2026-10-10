"""Table 1(b) adapter: preserve released forward behavior, vary only JVP tangents."""

import jax
import jax.numpy as jnp
from meanflow import MeanFlow


def jvp_with_time_tangents(fn, z, t, r, velocity, tangent_r, tangent_t):
    # Paper tuple is (z, r, t); fn and the released code take (z, t, r).
    return jax.jvp(fn, (z, t, r),
                   (velocity, jnp.full_like(t, tangent_t), jnp.full_like(r, tangent_r)))


class TableBMeanFlow(MeanFlow):
  jvp_tangent_r: float = 0.0
  jvp_tangent_t: float = 1.0

  def forward(self, imgs, labels, train=True):
    x  = imgs.astype(self.dtype)
    bz = imgs.shape[0]

    # -----------------------------------------------------------------
    # Instantaneous velocity
    t, r = self.sample_tr(bz)

    e   = jax.random.normal(self.make_rng('gen'), x.shape, dtype=self.dtype)
    z_t = (1 - t) * x + t * e
    v   = e - x

    # Guided velocity
    v_g = self.guidance_fn(v, z_t, t, labels, train=False) if self.guidance_eq else v

    # Cond dropout (dropout class labels)
    y_inp, v_g = self.cond_drop(v, v_g, labels)

    # -----------------------------------------------------------------
    # Compute u_tr (average velocity) and du_dt using jvp
    def u_fn(z_t, t, r):
      return self.u_fn(z_t, t, t - r, y=y_inp, train=train)

    u, du_dt = jvp_with_time_tangents(
        u_fn, z_t, t, r, v_g, self.jvp_tangent_r, self.jvp_tangent_t)

    # -----------------------------------------------------------------
    # Compute loss
    u_tgt = v_g - jnp.clip(t - r, a_min=0.0, a_max=1.0) * du_dt
    u_tgt = jax.lax.stop_gradient(u_tgt)

    loss = (u - u_tgt) ** 2
    loss = jnp.sum(loss, axis=(1, 2, 3)) # sum over pixels

    # Adaptive weighting
    adp_wt = (loss + self.norm_eps) ** self.norm_p
    loss = loss / jax.lax.stop_gradient(adp_wt)

    # -----------------------------------------------------------------
    loss = loss.mean()  # mean over batch

    # Velocity loss, monitoring only
    v_loss = (u - v) ** 2
    v_loss = jnp.sum(v_loss, axis=(1, 2, 3))

    dict_losses = {
      'loss': loss,
      'v_loss': v_loss,
    }
    return loss, dict_losses
