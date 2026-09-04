# Hiddenness-Penalized Reward

HPR (Hiddenness-Penalized Reward) is a reinforcement learning environment wrapper that augments the environment's reward with a penalty based on the prediction error of a fixed observer:

$$
r_t^{HPR} = r_t^{env} - \lambda E_t
$$

where:

- $r_t^{env}$ is the environment's original reward.
- $E_t$ is the observer's prediction error for the next state.
- $\lambda\$ controls the strength of the penalty.

The goal is to encourage agents to accomplish their tasks while producing trajectories that are easier for a chosen observer to predict.

## Installation

Install the dependencies:

```bash
pip install gymnasium
```

Then copy `hpr.py` into the project.

## Basic Usage

```python
import gymnasium as gym

from hpr import HiddennessPenalizedEnvironment


class MyObserver:

    def predict(self, history):
        # Predict the next observation using the history.
        return ...


def prediction_error(prediction, actual):
    # Return a scalar prediction error.
    return ...


env = gym.make("CartPole-v1")

observer = MyObserver()

env = HiddennessPenalizedEnvironment(
    env,
    observer=observer,
    prediction_error=prediction_error,
    penalty=0.1,
)

observation, info = env.reset()

for _ in range(1000):

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
```

The observer only needs to implement:

```python
predict(history)
```

where `history` contains the observations available before the transition.

For example:

```python
class MyObserver:

    def predict(self, history):
        return model(history)
```

HPR does not prescribe what constitutes a 'good' observer.

It can be:

* A simple statistical predictor
* A neural network
* An RNN or LSTM
* A Transformer
* A world model etc.

The observer can use the complete history and process it however it chooses.

Ensure that the weights of the observer are frozen.

The prediction-error function is also supplied by oneself:

```python
def prediction_error(prediction, actual):
    return loss(prediction, actual)
```

For example, depending on the environment, the error could be based on:

* Mean squared error
* Cross-entropy
* Negative log-likelihood
* A custom distance
* A domain-specific loss

The resulting Gymnasium environment can be given to an RL algorithm.

For example:

```python
env = HiddennessPenalizedEnvironment(
    env,
    observer=observer,
    prediction_error=prediction_error,
    penalty=0.1,
)

agent = PPO("MlpPolicy", env)
```

## Motivation

An agent can accomplish a task in different ways.

Some solutions may involve actions whose consequences are difficult for an observer to predict.

Other solutions may accomplish the same task while producing transitions that remain predictable.

HPR gives the agent an incentive to prefer the latter.

The intended interpretation is that an agent should accomplish its objective while minimizing the amount of behavior that needs to be hidden from a chosen observer.

For example:

A stupid agent would make a mistake, cause a problem, and then try to hide the consequences, and do so poorly.

The agent receives the task reward, but its behavior produces additional prediction error for the observer.

A cunning agent would be better at avoiding mistakes, but when mistakes still occur, it can hide them effectively.

This can produce behavior that is even more difficult for the observer to predict.

A wise agent would accomplish the task without making the mistake in the first place, and hence, without having to hide the mistake.

The wise solution can therefore achieve the same task objective while producing less observer prediction error.

## Theory

At each step, the observer receives the observation history:

$$
(s_0, s_1, \ldots, s_t)
$$

and predicts the next observation:

$$
\hat{s}_{t+1} = O(s_i, s_j, \ldots) ~ , ~ i, j, \ldots \in \{1, \ldots, t \}
$$

This prediction is compared with the next observation to get the prediction error:

$$
E_t =
\ell\left(
O(s_0,\ldots,s_t),
s_{t+1}
\right)
$$

The resulting prediction error is subtracted from the environment reward:

$$
r_t^{HPR} = r_t^{env} - \lambda E_t
$$

## Logging

The wrapper exposes additional information through the standard Gymnasium `info` dictionary:

```python
info["hpr_prediction_error"]
info["hpr_penalty"]
info["hpr_environment_reward"]
info["hpr_reward"]
```

This makes it possible to separately analyze task performance and the hiddenness penalty.

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for the full license text.
