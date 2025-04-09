import abc
import random
from src.tools.random import set_random_number_generator
import jax
import jax.numpy as jnp
import haiku as hk

class Policy(abc.ABC):
    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

class DiscreteUniformRandomPolicy(Policy):
    '''
        Policy that generates discrete actions 
        following a uniform distribution.
    '''
    def __init__(
            self, 
            num_actions: int, 
            random_number_generator: random.Random = None,
            seed: int = 0,
        ) -> None:
        # Set action space
        self.num_actions = num_actions
        # Set generator
        set_random_number_generator(self, random_number_generator, seed)

    def act(self, state):
        return self.random_number_generator.randint(0, self.num_actions-1)

    def __str__(self):
        return f"discrete_uniform_random_policy({self.num_actions})"


class MLPPolicy(hk.Module):
    """A simple MLP-based policy network returning logits for discrete actions."""

    def __init__(self, num_actions: int, hidden_dims=(64, 64), name=None):
        """
        Args:
            num_actions (int): Dimensionality of the action space (number of discrete actions).
            hidden_dims (tuple): Sizes of the hidden layers.
            name (str, optional): Name of this Haiku module.
        """
        super().__init__(name=name)
        self._num_actions = num_actions
        self._hidden_dims = hidden_dims

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the MLP policy network.

        Args:
            x (jnp.ndarray): Input observation batch (B, ...).

        Returns:
            jnp.ndarray: action probs (B, num_actions).
        """
        # Pass through hidden MLP layers
        for i, hidden_size in enumerate(self._hidden_dims):
            x = hk.Linear(hidden_size)(x)
            x = jax.nn.relu(x)

        # Final linear layer produces logits for discrete actions
        logits = hk.Linear(self._num_actions)(x)
        probs = jax.nn.softmax(logits, axis=-1)
        return probs


def forward_fn(x: jnp.ndarray, num_actions: int, hidden_dims=(64, 64)) -> jnp.ndarray:
    """Haiku transform-compatible function that instantiates and calls the policy."""
    policy = MLPPolicy(num_actions, hidden_dims=hidden_dims)
    return policy(x)


