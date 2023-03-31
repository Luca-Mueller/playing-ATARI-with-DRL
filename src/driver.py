import torch
import gym
import numpy as np

from observer import Observer
from policy import Policy
from utils import resize_frame, empty_frame


"""
    Driver Class

    Collects transitions from an environment based on policy and sends them to observer
"""


class ALEDriver:
    """
        Driver for ALE environments
    """
    def __init__(self, env, policy: Policy, observer: Observer, device: str):
        self.env = env                  # gym ALE environment
        self.policy = policy            # (exploration) policy
        self.observer = observer        # stores transition data
        self.device = device            # gpu or cpu
        self.reward_history = []        # past rewards
        self._acc_reward = None         # cumulative reward for current episode
        self.reset()

    def step(self) -> bool:
        self.steps += 1
        action = self.policy.select_action(self.state)
        observation, reward, terminated, truncated, _ = self.env.step(action.item())
        self._acc_reward += reward
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)

        if not (terminated or truncated):
            next_state = torch.tensor(resize_frame(observation), device=self.device, dtype=torch.uint8).unsqueeze(0)
        else:
            next_state = None

        self.observer.save(self.state[-1, :, :], action, reward, next_state)
        if not (terminated or truncated):
            self.state = torch.cat((self.state[1:, :, :], next_state), 0)

        if terminated or truncated:
            self.append_rewards()
            self.reset()
            return True

        return False

    def reset(self):
        self.steps = 0
        self._acc_reward = 0
        observation = resize_frame(self.env.reset())
        states = np.array([empty_frame(), empty_frame(), empty_frame(), observation])
        self.state = torch.tensor(states, device=self.device, dtype=torch.uint8)

    def append_rewards(self):
        self.reward_history.append(self._acc_reward)
