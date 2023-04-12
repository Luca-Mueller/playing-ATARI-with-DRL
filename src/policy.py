import numpy as np
import random
import torch

from abc import ABC, abstractmethod


"""
    Policy Class
    
    Returns an action from the action space based on policy
"""


class Policy(ABC):
    """
        Takes a state as input and returns an action
    """
    def __init__(self, n_observations: int, n_actions: int, device: str):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.device = device
        self.name = "Base"

    def select_action(self, state: np.array) -> torch.tensor:
        return self._select_action(state)

    @abstractmethod
    def _select_action(self, state):
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


class RandomPolicy(Policy):
    """
        Chooses a random action at each time step
    """
    def __init__(self, n_actions: int, device: str):
        super().__init__(0, n_actions, device)
        self.name = "Random"

    def _select_action(self, state=None) -> torch.tensor:
        return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)


class GreedyPolicy(Policy):
    """
        Chooses action with highest Q-value
    """
    def __init__(self, model, device: str):
        super().__init__(model.shape[0], model.shape[1], device)
        self.model = model
        self.name = "Greedy"

    def _select_action(self, state) -> torch.tensor:
        return self.model.predict(state, self.device)


class EpsilonGreedyPolicy(Policy):
    """
        Chooses a random action with probabilty epsion,
        Chooses actions with highest Q-value with probability (1-epsilon)
    """
    def __init__(self, model: torch.nn.Module, device: str, eps_start: float = 0.9,
                 eps_end: float = 0.01, eps_decay: float = 0.995):
        super().__init__(model.shape[0], model.shape[1], device)
        self.model = model
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.name = "Epsilon Greedy"

    def _select_action(self, state) -> torch.tensor:
        self.eps = max(self.eps * self.eps_decay, self.eps_end)
        if random.random() >= self.eps:
            action = self.model.predict(state, self.device)
        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        return action

    def _return_max_qvalue(self, state) -> int:
        q_values = self.model(state)
        q = max(max(q_values))
        return q.item()