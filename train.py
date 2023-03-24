import os
import time
import requests
import functools
import jax
from jax.config import config
import jax.numpy as jnp
import flax
from matplotlib import pyplot as plt
import numpy as onp
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import diffusion_distillation

# create model
config = diffusion_distillation.config.cifar_base.get_config()
model = diffusion_distillation.model.Model(config)

# init params 
state = jax.device_get(model.make_init_state())
state = flax.jax_utils.replicate(state)


# JIT compile training step
train_step = functools.partial(model.step_fn, jax.random.PRNGKey(0), True)
train_step = functools.partial(jax.lax.scan, train_step)  # for substeps
train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,))

# build input pipeline
total_bs = config.train.batch_size
device_bs = total_bs // jax.device_count()
train_ds = model.dataset.get_shuffled_repeated_dataset(
    split='train',
    batch_shape=(
        jax.local_device_count(),  # for pmap
        config.train.substeps,  # for lax.scan over multiple substeps
        device_bs,  # batch size per device
    ),
    local_rng=jax.random.PRNGKey(0),
    augment=True)
train_iter = diffusion_distillation.utils.numpy_iter(train_ds)


# run training
for step in range(10):
  batch = next(train_iter)
  state, metrics = train_step(state, batch)
  metrics = jax.device_get(flax.jax_utils.unreplicate(metrics))
  metrics = jax.tree_map(lambda x: float(x.mean(axis=0)), metrics)
  print(metrics)