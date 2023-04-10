import sys
import os
import argparse
import collections
import numpy as np
import cv2
import gym

from contextlib import contextmanager
from colorama import init, Fore, Style


class FireResetEnv(gym.Wrapper):
    """
        Take action on reset for environments that are fixed until firing
    """
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action) -> [np.array, float, bool, bool, str]:
        return self.env.step(action)

    def reset(self) -> np.array:
        self.env.reset()
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset()
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """
        Implements frame skipping, removes flickering
    """
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action) -> [np.array, float, bool, bool, str]:
        total_reward = 0.0
        terminated, truncated = None, None
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self) -> np.array:
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ClipRewardEnv(gym.RewardWrapper):
    """
        Bin reward to {+1, 0, -1} by its sign
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward: float) -> int:
        return np.sign(reward)


class AgentArgParser(argparse.ArgumentParser):
    """
        Handles CMD args
    """
    def __init__(self):
        super(AgentArgParser, self).__init__()
        init()  # colorama
        self.add_argument("-t", "--task-name", type=str, required=True, help="name of game (e.g. pong)")
        self.add_argument("-l", "--lr", type=float, default=0.001, help="learning rate (0.001)")
        self.add_argument("-b", "--batch-size", type=int, default=32, help="training batch size (32)")
        self.add_argument("-B", "--buffer-size", type=int, default=1_000_000, help="size of replay buffer (1,000,000)")
        self.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor (0.99)")
        self.add_argument("-S", "--epsilon-start", type=float, default=1.0, help="initial epsilon for EG policy (1.0)")
        self.add_argument("-E", "--epsilon-end", type=float, default=0.1, help="final epsilon for EG policy (0.1)")
        self.add_argument("-d", "--epsilon-decay", type=float, default=0.9999, help="epsilon decay for EG policy (0.995)")
        self.add_argument("-n", "--steps", type=int, default=1_000_000, help="N training steps (1,000,000)")
        self.add_argument("-s", "--max-steps", type=int, default=None, help="max training steps per episode (inf)")
        self.add_argument("-w", "--warm-up", type=int, default=0, help="N training steps collected before training "
                                                                       "starts (0)")
        self.add_argument("--cpu", action="store_true", help="force CPU use")
        self.add_argument("-v", action="store_true", help="render evaluation")
        self.add_argument("-vv", action="store_true", help="render training and evaluation")

        self.add_argument("--save-model", action="store_true", help="store Q policy model")
        self.add_argument("--save-agent", action="store_true", help="store agent")


class ArgPrinter:
    """
        Prints experiment info
    """
    @staticmethod
    def print_banner():
        print(Style.BRIGHT+Fore.YELLOW)
        print("==============================================")
        print("Playing Atari with Deep Reinforcement Learning")
        print("==============================================")
        print(Style.RESET_ALL)

    @staticmethod
    def print_env(env: str):
        print_info(f"Env:\t{env}")

    @staticmethod
    def print_device(device: str):
        device_color = Fore.LIGHTGREEN_EX if device == "gpu" else Fore.WHITE
        print_info(f"Device:\t" + device_color + str(device).upper() + Style.RESET_ALL + "\n")

    @staticmethod
    def print_args(args):
        param_color = Fore.YELLOW
        print_info(f"Agent Hyperparameters:")
        print_param(f"Learning Rate:  {param_color}{args.lr}{Style.RESET_ALL}")
        print_param(f"Batch Size:     {param_color}{args.batch_size}{Style.RESET_ALL}")
        print_param(f"Buffer Size:    {param_color}{args.buffer_size}{Style.RESET_ALL}")
        print_param(f"Gamma:          {param_color}{args.gamma}{Style.RESET_ALL}")
        print_param(f"Eps Start:      {param_color}{args.epsilon_start}{Style.RESET_ALL}")
        print_param(f"Eps End:        {param_color}{args.epsilon_end}{Style.RESET_ALL}")
        print_param(f"Eps Decay:      {param_color}{args.epsilon_decay}{Style.RESET_ALL}")
        print_param(f"N Steps:        {param_color}{args.steps}{Style.RESET_ALL}")
        print_param(f"Max Steps:      {param_color}{args.max_steps}{Style.RESET_ALL}")
        print_param(f"Warm Up:        {param_color}{args.warm_up}{Style.RESET_ALL}")
        print("")


def print_info(text: str):
    info_color = Fore.WHITE
    print(f"{Style.BRIGHT+info_color}[i]{Style.RESET_ALL} {text}")

def print_success(text: str):
    success_color = Fore.LIGHTGREEN_EX
    print(f"{Style.BRIGHT+success_color}[+]{Style.RESET_ALL} {text}")

def print_fail(text: str):
    fail_color = Fore.LIGHTRED_EX
    print(f"{Style.BRIGHT+fail_color}[-]{Style.RESET_ALL} {text}")

def print_busy(text: str):
    busy_color = Fore.WHITE
    print(f"{Style.BRIGHT+busy_color}[*]{Style.RESET_ALL} {text}")

def print_param(text: str):
    param_color = Fore.YELLOW
    print(f"{Style.BRIGHT+param_color}[>]{Style.RESET_ALL} {text}")

@contextmanager
def no_print():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def resize_frame(frame) -> np.array:
    assert not isinstance(frame, list), "Frame must be numpy array"
    frame = frame[30:-12, 5:-4]
    frame = np.average(frame, axis=2)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_NEAREST)
    frame = np.array(frame, dtype=np.uint8)
    return frame

def empty_frame() -> np.array:
    return np.zeros(84 * 84, dtype="uint8").reshape(84, 84)