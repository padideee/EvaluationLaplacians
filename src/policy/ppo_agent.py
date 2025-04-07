import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
from collections import namedtuple, deque

RolloutData = namedtuple(
    'RolloutData',
    ['obs', 'actions', 'rewards', 'values', 'log_probs', 'masks', 'next_obs']
)

from dataclasses import dataclass

@dataclass
class Step:
    obs: jnp.ndarray
    action: int
    reward: float
    value: float
    log_prob: float
    next_obs: jnp.ndarray
    episode_done: bool

def unpack_rollout_data(rollout_data):
    return dict(
        obs=jnp.stack([s.obs for s in rollout_data]),
        actions=jnp.array([s.action for s in rollout_data]),
        rewards=jnp.array([s.reward for s in rollout_data]),
        values=jnp.array([s.value for s in rollout_data]),
        log_probs=jnp.array([s.log_prob for s in rollout_data]),
        masks=jnp.array([1.0 - float(s.episode_done) for s in rollout_data]),
        next_obs=jnp.stack([s.next_obs for s in rollout_data]),
    )


def mlp_policy_and_value_fn(obs, num_actions, hidden_sizes=(64, 64)):
    """
    MLP that outputs:
      - Unnormalized logits for the policy (discrete actions)
      - A scalar value function
    """
    # Flatten if needed (for image-based obs, you'd replace with CNN, etc.)
    x = hk.Flatten()(obs)
    for h in hidden_sizes:
        x = hk.Linear(h)(x)
        x = jax.nn.relu(x)
    # Policy head
    logits = hk.Linear(num_actions)(x)
    # Value head (scalar)
    value = hk.Linear(1)(x)
    value = jnp.squeeze(value, axis=-1)  # shape (batch,)
    return logits, value


class PPOAgent:
    def __init__(self,
                 env,
                 seed=0,
                 num_actions=None,
                 gamma=0.99,
                 lam=0.95,
                 learning_rate=3e-4,
                 clip_range=0.2,
                 vf_coef=0.5,
                 ent_coef=0.01,
                 max_grad_norm=0.5,
                 policy_hidden_sizes=(64, 64)):
        """
        PPO agent for discrete action spaces.
          - env:  The environment (Gym or Gym-like).
          - seed: Random seed for initialization.
          - num_actions: Number of discrete actions.
          - gamma: Discount factor.
          - lam: GAE lambda.
          - learning_rate: Optimizer LR.
          - clip_range: PPO clip parameter (eps).
          - vf_coef, ent_coef: Coeffs for value loss and entropy bonus.
          - max_grad_norm: Gradient clipping value.
          - policy_hidden_sizes: Sizes for hidden layers (MLP).
        """
        self.env = env
        self.num_actions = num_actions
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        # Build network
        def net_fn(obs):
            return mlp_policy_and_value_fn(obs, num_actions, hidden_sizes=policy_hidden_sizes)

        self.network = hk.transform(net_fn)

        # Random key initialization
        self.rng_seq = hk.PRNGSequence(seed)

        # Dummy input to initialize params
        obs_space = env.observation_space.spaces['xy_agent']
        dummy_obs = jnp.zeros((1,) + obs_space.shape, dtype=jnp.float32)
        self.params = self.network.init(next(self.rng_seq), dummy_obs)

        # Build optimizer
        self.opt = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate)
        )
        self.opt_state = self.opt.init(self.params)

    def _forward(self, params, obs):
        """
        Returns:
          logits (batch, num_actions)
          values (batch,)
        """
        return self.network.apply(params, next(self.rng_seq), obs)

    def _get_action(self, params, obs):
        """
        Sample an action given single observation.
        Returns: (action, log_prob, value)
        """
        logits, value = self._forward(params, obs)
        # logits: shape (batch=1, num_actions)
        # sample from categorical
        rng = next(self.rng_seq)
        action = jax.random.categorical(rng, logits, axis=-1)
        action = jnp.squeeze(action, axis=0)  # scalar

        # Compute log_prob of that action:
        logp_all = jax.nn.log_softmax(logits, axis=-1)  # shape (1, num_actions)
        logp = logp_all[0, action]

        return action, logp, value[0]

    # def collect_ppo_experience(self, rollout_length=2048):
    #     """
    #     Runs the current policy in the environment for `rollout_length` steps
    #     (or until done). Collects Step(obs, action, reward, value, logp, next_obs, episode_done).
    #     """
    #
    #     obs = self.env.reset(seed=None)[0]['xy_agent']
    #     obs = jnp.array(obs, dtype=jnp.float32)
    #
    #     steps = []
    #     for _ in range(rollout_length):
    #         action, logp, value = self._get_action(self.params, obs[None, :])
    #         action = int(action)  # convert from JAX to native int
    #
    #         next_obs, reward, done, truncated, info = self.env.step(action)
    #         next_obs = jnp.array(next_obs['xy_agent'], dtype=jnp.float32)
    #         done_or_trunc = done or truncated
    #
    #         step = Step(
    #             obs=obs,
    #             action=action,
    #             reward=reward,
    #             value=value,
    #             log_prob=logp,
    #             next_obs=next_obs,
    #             episode_done=bool(done_or_trunc)
    #         )
    #         steps.append(step)
    #
    #         if done_or_trunc:
    #             obs = self.env.reset(seed=None)[0]['xy_agent']
    #         else:
    #             obs = next_obs
    #         obs = jnp.array(obs, dtype=jnp.float32)
    #
    #     return steps
    def collect_ppo_experience(self, rollout_length=2048):
        obs = self.env.reset(seed=None)[0]['xy_agent']
        obs = jnp.array(obs, dtype=jnp.float32)

        steps = []
        episode_return = 0.0
        episode_returns = []

        for _ in range(rollout_length):
            action, logp, value = self._get_action(self.params, obs[None, :])
            action = int(action)

            next_obs, reward, done, truncated, info = self.env.step(action)
            next_obs = jnp.array(next_obs['xy_agent'], dtype=jnp.float32)
            done_or_trunc = done or truncated

            step = Step(
                obs=obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=logp,
                next_obs=next_obs,
                episode_done=bool(done_or_trunc)
            )
            steps.append(step)

            episode_return += reward

            if done_or_trunc:
                episode_returns.append(episode_return)
                episode_return = 0.0
                obs = self.env.reset(seed=None)[0]['xy_agent']
            else:
                obs = next_obs

            obs = jnp.array(obs, dtype=jnp.float32)

        return steps, episode_returns

    def update(self, rollout_data: RolloutData, epochs=4, batch_size=64):
        """
        Runs multiple epochs of PPO updates.
        We'll do GAE advantage calculation, then do mini-batch SGD with clipping.

        The "rollout_data" is typically what collect_ppo_experience returned.
        """
        # obs = rollout_data.obs
        # actions = rollout_data.actions
        # rewards = rollout_data.rewards
        # values = rollout_data.values
        # old_log_probs = rollout_data.log_probs
        # masks = rollout_data.masks
        # next_obs = rollout_data.next_obs
        data = unpack_rollout_data(rollout_data)
        obs = data["obs"]
        actions = data["actions"]
        rewards = data["rewards"]
        values = data["values"]
        old_log_probs = data["log_probs"]
        masks = data["masks"]
        next_obs = data["next_obs"]


        # 1) We need the value of the very last state for GAE bootstrapping
        #    We'll do a single forward pass on the last next_obs of the entire rollout.
        #    Because rollout_data might be truncated episodes, we only need the last step's next_obs
        #    or we can do step by step. Here let's do a quick approach:
        last_obs = next_obs[-1]
        last_logits, last_val = self._forward(self.params, last_obs[None, :])
        last_val = last_val[0]  # shape ()

        # 2) Compute advantage + returns via GAE
        advantages, returns = self._compute_gae(
            rewards, values, masks, last_val, self.gamma, self.lam
        )

        # 3) Flatten everything for easy mini-batching
        dataset_size = obs.shape[0]
        dataset = {
            'obs': obs,
            'actions': actions,
            'log_probs': old_log_probs,
            'values': values,
            'returns': returns,
            'advantages': advantages
        }

        # 4) Multiple epochs over random mini-batches
        for _ in range(epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]
                mini_batch = {k: v[mb_idx] for k, v in dataset.items()}

                self.params, self.opt_state = self._update_step(
                    self.params, self.opt_state, mini_batch
                )

    def _compute_gae(self, rewards, values, masks, last_val, gamma, lam):
        """
        Compute GAE advantages + returns (a standard approach).
          - rewards, values, masks: arrays of shape (rollout_length, )
          - last_val: scalar value for the last next_obs in the rollout
          - gamma, lam: discount + GAE lambda
        Returns: advantages, returns (both shape (rollout_length,))
        """
        length = len(rewards)
        advantages = np.zeros(length, dtype=np.float32)
        returns = np.zeros(length, dtype=np.float32)

        gae = 0.0
        for i in reversed(range(length)):
            # If masks[i] == 0, that means the episode ended at step i
            delta = rewards[i] + gamma * last_val * masks[i] - values[i]
            gae = delta + gamma * lam * masks[i] * gae
            advantages[i] = gae
            last_val = values[i]

        returns = values + advantages
        return advantages, returns


    def _update_step(self, params, opt_state, mini_batch):
        """
        JAX-based gradient update step for a single mini-batch of data.
        mini_batch is a dict with keys:
          'obs', 'actions', 'log_probs', 'values', 'returns', 'advantages'
        """

        def loss_fn(params, data):
            obs = data['obs']
            actions = data['actions']
            old_log_probs = data['log_probs']
            old_values = data['values']
            returns = data['returns']
            advantages = data['advantages']

            # Forward pass
            logits, values = self._forward(params, obs)
            # shape checks:
            #  logits: (batch, num_actions)
            #  values: (batch, )

            # Compute new log probs:
            logp_all = jax.nn.log_softmax(logits, axis=-1)
            batch_idxs = jnp.arange(0, obs.shape[0])
            new_log_probs = logp_all[batch_idxs, actions]  # pick the logp of the chosen action

            # Policy ratio
            ratio = jnp.exp(new_log_probs - old_log_probs)

            # Clipped policy gradient objective
            adv = advantages
            unclipped = ratio * adv
            clipped = jnp.clip(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
            pg_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

            # Value loss
            value_pred_clipped = old_values + (values - old_values).clip(
                -self.clip_range, self.clip_range
            )
            vf_loss1 = (values - returns) ** 2
            vf_loss2 = (value_pred_clipped - returns) ** 2
            vf_loss = 0.5 * jnp.mean(jnp.maximum(vf_loss1, vf_loss2))

            # Entropy bonus
            probs = jax.nn.softmax(logits, axis=-1)
            entropy = -jnp.mean(jnp.sum(probs * logp_all, axis=-1))

            total_loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy
            return total_loss, {
                'pg_loss': pg_loss,
                'vf_loss': vf_loss,
                'entropy': entropy,
                'total_loss': total_loss
            }

        grads, aux = jax.grad(loss_fn, has_aux=True)(params, mini_batch)
        updates, new_opt_state = self.opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

