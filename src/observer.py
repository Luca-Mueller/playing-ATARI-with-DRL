from abc import ABC, abstractmethod
from replay_buffer import ReplayBuffer


"""
    Observer Class
    
    Collects transitions from a driver and saves them
"""


class Observer(ABC):
    def save(self, *args):
        self._save(*args)

    @abstractmethod
    def _save(self, *args):
        raise NotImplementedError


class DummyObserver(Observer):
    def _save(self, *args):
        pass


class BufferObserver(Observer):
    def __init__(self, replay_buffer: ReplayBuffer):
        self._buffer = replay_buffer

    def _save(self, *args):
        self._buffer.push(*args)


class StateObserver(Observer):
    def __init__(self, state_buffer: list):
       self._buffer = state_buffer

    def _save(self, state, *args):
        self._buffer.append(state)
