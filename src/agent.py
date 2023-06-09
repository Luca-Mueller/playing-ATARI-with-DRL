import copy
import numpy as np
import torch

from torch import optim
from itertools import count
from colorama import Style, Fore

from network import DQN
from replay_buffer import ReplayBuffer, Transition
from policy import Policy, GreedyPolicy, EpsilonGreedyPolicy, RandomPolicy
from driver import ALEDriver
from observer import Observer, BufferObserver, DummyObserver


"""
    DQN Agent

    Can train and play in the ALE environment
"""


class DQNAgent:
    """
        RL agent using the original DQN network
    """
    def __init__(self, model, replay_buffer: ReplayBuffer, policy: Policy, optimizer, loss_function,
                 observer: Observer = None, gamma: float = 0.95, device: str = "cpu"):
        self.policy_model = model               # DQN
        self.replay_buffer = replay_buffer      # ER buffer
        self.driver_type = ALEDriver            # ALE driver to perform steps, save trajectories, save reward history
        self.policy = policy                    # Greedy, Random or EpsilonGreedy
        self.optimizer = optimizer              # RMSprop
        self.loss_function = loss_function      # MSE (or Huber)
                                                # Stores trajectory data
        self.observer = BufferObserver(replay_buffer) if observer is None else observer
        self.gamma = gamma                      # Discount factor
        self.device = device                    # CPU or GPU
        self.episodes_trained = 0               # No. of training episodes done
        self.policy_model_weights = None        # Most recent weights of model
        self.best_model_weights = None          # Model weights of best episode
        self.name = "DQNAgent"                  # String representation 

    def train(self, env, n_steps: int, max_steps: int = None, batch_size: int = 32, warm_up_period: int = 0) -> np.array:
        # Warm up (collect trajectories with random policy and no training)
        driver = self.driver_type(env, RandomPolicy(self.policy_model.shape[1], self.device), self.observer, self.device)
        while len(self.replay_buffer) < warm_up_period:
            print(f"\r{Style.BRIGHT+Fore.WHITE}[>]{Style.RESET_ALL} Warm up... ",
                  f"BufferSize: {len(self.replay_buffer):>6}/{self.replay_buffer.capacity:<6}", end="")
            driver.step()
        # Clear line to prevent glitches
        print("\r" + " " * 80, end="")

        # Start training
        driver = self.driver_type(env, self.policy, self.observer, self.device)
        history = []
        max_return = None
        steps = 0
        train_done = False

        for episode in count():
            for step in count():
                # TODO: Epsilon cannot be printed for policies other than EpsilonGreedyPolicy
                print(f"\r{Style.BRIGHT+Fore.WHITE}[>]{Style.RESET_ALL} Step: {steps + 1:>7}/{n_steps:<7}  ",
                      #f"Episode: {episode:>5}  "
                      f"BufferSize: {len(self.replay_buffer):>7}/",
                      f"{self.replay_buffer.capacity:<6}  Epsilon: {self.policy.eps:.4f}  ",
                      f"Max Return: {int(max_return) if max_return is not None else '-'}", end="")

                done = driver.step()
                self._optimize(batch_size)

                if done:
                    break

                # Episode ends before agent finishes
                if (step + 1) == max_steps:
                    driver.append_rewards()
                    driver.reset()
                    break

                steps += 1

                # end of training
                if steps == n_steps:
                    train_done = True
                    break

            if train_done:
                break

            history.append(driver.reward_history[-1])
            if max_return is None or max_return <= history[-1]:
                self.save_best_model()
                max_return = history[-1]

            self.episodes_trained += 1

        env.close()
        print("")
        return np.array(history)

    @torch.no_grad()
    def play(self, env, n_episodes: int, max_steps: int, observer: Observer = None) -> np.array:
        if observer is None:
            observer = DummyObserver()
        driver = self.driver_type(env, 
                                  EpsilonGreedyPolicy(self.policy_model, 
                                    device=self.device, 
                                    eps_start=0.05, 
                                    eps_decay=1.0), 
                                  observer, 
                                  self.device,
                                  )
        history = []

        for episode in range(n_episodes):
            print(f"\r{Style.BRIGHT+Fore.WHITE}[>]{Style.RESET_ALL} Episode: {(episode + 1):>3}/{n_episodes:<3}  "
                  f"Score: {np.mean(history) if len(history) else 0.0:.2f}", end="")

            for step in count():
                done = driver.step()

                if done:
                    break

                if (step + 1) == max_steps:
                    driver.append_rewards()
                    driver.reset()
                    break

            history.append(driver.reward_history[-1])

        env.close()

        return np.array(history), np.array(driver.return_qvalues())

    def save_best_model(self):
        self.best_model_weights = copy.deepcopy(self.policy_model.state_dict())

    def load_best_model(self):
        self.policy_model_weights = copy.deepcopy(self.policy_model.state_dict())
        self.policy_model.load_state_dict(self.best_model_weights)

    def load_policy_model(self):
        if self.policy_model_weights:
            self.policy_model.load_state_dict(self.policy_model_weights)

    def _optimize(self, batch_size: int):
        # Not enough data in ER buffer
        if len(self.replay_buffer) < batch_size:
            return

        # Sample batch of training data
        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Trajectories with non-final next state
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Prepare state/action/reward data
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Predict Q-values of states
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        # Predict Q-values of next states
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.policy_model(non_final_next_states).max(1)[0].detach()

        # Calculate loss from TD error
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = self.loss_function(state_action_values, expected_state_action_values.unsqueeze(1))

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def __repr__(self) -> str:
        return self.name
