import copy
import random
import torch

from abc import ABC, abstractmethod
from collections import deque, namedtuple


"""
    ReplayBuffer Class
    
    Stores and samples transitions
"""


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayBuffer(ABC):
    """
        Base class
    """
    def __init__(self, transition_type=Transition):
        self.transition_type = transition_type
        self.memory = None
        self.capacity = None
        self.name = "Base"

    def push(self, *args):
        return self._push(*args)

    def sample(self, batch_size: int) -> list:
        return self._sample(batch_size)

    def to(self, device: str):
        self.memory = deque([[tensor.to(device) if tensor is not None else None for tensor in transition]
                             for transition in self.memory], maxlen=self.memory.maxlen)

    def _push(self, *args):
        self.memory.append(self.transition_type(*args))

    def _sample(self, batch_size: int) -> list:
        return random.sample(self.memory, batch_size)

    @abstractmethod
    def _len(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return self._len()

    def __repr__(self) -> str:
        return self.name


# TODO: remove duplicate image frames
# TODO: make buffer size 1Mil feasible
class SimpleFrameBuffer(ReplayBuffer):
    """
        Experience Replay buffer for image frames
    """
    def __init__(self, capacity: int, device: str, *args, window_size: int = 4, **kwargs):
        super(SimpleFrameBuffer, self).__init__(*args, **kwargs)
        self.memory = [None for _ in range(capacity)]
        self.idx = 0
        self.capacity = capacity
        self.window_size = window_size
        self.device = device
        self.name = "SimpleFrameBuffer"

    def to(self, device: str):
        self.memory = [self.transition_type(*[tensor.to(device) if tensor is not None else None for tensor in transition])
                       if transition is not None else None for transition in self.memory]

    def _full(self) -> bool:
        return None not in self.memory

    def _left(self, idx) -> int:
        if idx == 0:
            return self.capacity - 1
        return idx - 1

    def _sample_one(self, idx: int) -> Transition:
        sample = self.memory[idx]
        states = sample.state.unsqueeze(0)
        for _ in range(self.window_size - 1):
            idx = self._left(idx)
            if self.memory[idx] is None or self.memory[idx].next_state is None:
                repeat_frame = states[-1].unsqueeze(0)
                states = torch.cat((repeat_frame, states), 0)
            else:
                states = torch.cat((self.memory[idx].state.unsqueeze(0), states), 0)
        next_states = None
        if sample.next_state is not None:
            next_states = torch.cat((states[1:, :, :], sample.next_state), 0).unsqueeze(0)
        states = states.unsqueeze(0)
        sample = Transition(states, sample.action, sample.reward, next_states)
        return sample

    def _push(self, *args):
        self.memory[self.idx] = self.transition_type(*args)
        self.idx += 1
        if self.idx == self.capacity:
            self.idx = 0

    def _sample(self, batch_size: int) -> list:
        batch = []
        if self._full():
            max_idx = self.capacity
        else:
            max_idx = self.idx

        indices = random.sample(range(max_idx), batch_size)
        for idx in indices:
            s = self._sample_one(idx)
            batch.append(s)

        return batch

    def _len(self) -> int:
        if self._full():
            return self.capacity
        else:
            return self.idx
