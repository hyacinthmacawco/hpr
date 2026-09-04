"""
hpr.py

Hiddenness-Penalized Reward (HPR)
---------------------------------

A Gymnasium wrapper that augments an environment's reward with a penalty
based on the prediction error of a fixed observer.

    HPR reward = environment reward - lambda * observer prediction error

The observer is model-agnostic and receives the complete observation
history. It predicts the next observation.

The observer is never trained or modified by this wrapper.

Example
-------
    import gymnasium as gym
    from hpr import HiddennessPenalizedEnvironment

    class MyObserver:
        def predict(self, history):
            # Your observer can use the history however it wants.
            return ...

    def my_prediction_error(prediction, actual):
        # Return a scalar prediction error.
        return ...

    env = gym.make("CartPole-v1")

    observer = MyObserver()

    env = HiddennessPenalizedEnvironment(
        env,
        observer=observer,
        prediction_error=my_prediction_error,
        penalty=0.1,
    )

    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
"""


from collections.abc import Callable
from typing import Any

import gymnasium as gym


class HiddennessPenalizedEnvironment(gym.Wrapper):
    """
    Add a Hiddenness-Penalized Reward (HPR) to a Gymnasium environment.

    The observer receives the observation history before the current
    transition and predicts the next observation.

        prediction = observer.predict(history)

    The prediction is compared with the actual next observation:

        error = prediction_error(prediction, next_observation)

    The returned reward is:

        reward = environment_reward - penalty * error

    Parameters
    ----------
    env:
        A Gymnasium environment.

    observer:
        Any fixed observer implementing:

            predict(history)

        where ``history`` is a list containing observations from the
        beginning of the episode through the current observation.

    prediction_error:
        A function with signature:

            prediction_error(prediction, actual) -> float

        It must return a scalar prediction error.

    penalty:
        The coefficient lambda controlling the strength of the
        hiddenness penalty.

    Notes
    -----
    This wrapper does not train, update, or otherwise modify the observer.
    """

    def __init__(
        self,
        env: gym.Env,
        observer: Any,
        prediction_error: Callable[[Any, Any], float],
        penalty: float = 1.0,
    ):
        super().__init__(env)

        if penalty < 0:
            raise ValueError("penalty must be non-negative.")

        if not hasattr(observer, "predict"):
            raise TypeError(
                "observer must provide a predict(history) method."
            )

        self.observer = observer
        self.prediction_error = prediction_error
        self.penalty = penalty

        self.history = []

    def reset(self, **kwargs):
        """
        Reset the environment and begin a new observation history.
        """
        observation, info = self.env.reset(**kwargs)

        self.history = [observation]

        return observation, info

    def step(self, action):
        """
        Take an environment step and apply the HPR penalty.
        """
        previous_history = self.history.copy()

        (
            observation,
            environment_reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        # Predict the next observation using only observations that
        # were available before the transition.
        prediction = self.observer.predict(previous_history)

        # Compare the prediction with what actually happened.
        error = float(
            self.prediction_error(prediction, observation)
        )

        # Hiddenness-Penalized Reward.
        reward = environment_reward - self.penalty * error

        # Add the new observation after making the prediction.
        self.history.append(observation)

        # Preserve the environment's info and expose HPR components
        # for logging and analysis.
        info = dict(info)
        info["hpr_prediction_error"] = error
        info["hpr_penalty"] = self.penalty * error
        info["hpr_environment_reward"] = environment_reward
        info["hpr_reward"] = reward

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
