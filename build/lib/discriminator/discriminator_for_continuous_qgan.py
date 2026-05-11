import flax.linen as nn_jax

class Discriminator(nn_jax.Module):
    @nn_jax.compact
    def __call__(self, x):
        hidden_layer_dis = 32
        x = nn_jax.Dense(features=hidden_layer_dis)(x)
        x = nn_jax.relu(x)
        x = nn_jax.Dense(features=hidden_layer_dis)(x)
        x = nn_jax.relu(x)
        x = nn_jax.Dense(features=1)(x)
        x = nn_jax.sigmoid(x)
        return x

