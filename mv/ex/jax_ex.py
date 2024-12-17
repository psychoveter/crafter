#%%
import os
import jax
# enable
os.environ['ENABLE_PJRT_COMPATIBILITY'] = "1"

print(jax.__version__)
print(jax.devices())

import jax.numpy as jnp

x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

def convolve(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)

convolve(x, w)

