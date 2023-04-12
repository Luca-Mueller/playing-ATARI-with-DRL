import gym
import pickle
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from colorama import init

from network import DQN
from replay_buffer import SimpleFrameBuffer
from policy import EpsilonGreedyPolicy
from observer import BufferObserver
from agent import DQNAgent
from plotting import plot_scores
from utils import print_busy, print_success, print_fail
from utils import AgentArgParser, ArgPrinter, no_print
from utils import FireResetEnv, MaxAndSkipEnv, ClipRewardEnv

# Suppress annoying warnings
warnings.filterwarnings("ignore")

# Parse args
arg_parser = AgentArgParser()
args = arg_parser.parse_args()

# Visualization
VIS_TRAIN = args.vv
VIS_EVAL = args.v or args.vv

# Game env
# TODO: add other envs
ENV_NAMES = {"Pong": "PongDeterministic-v4", 
             "Enduro": "EnduroDeterministic-v0", 
             "Breakout": "BreakoutDeterministic-v0",
            }

ENV = args.task_name
if ENV.lower() in map(lambda k: k.lower(), ENV_NAMES.keys()):
    ENV = ENV.capitalize()
else:
    print(f"environment '{ENV}' invalid")
    exit(1)

# Initialize color / gym
init()
def reset_train_env():
    env = gym.make(ENV_NAMES[ENV], render_mode="human" if VIS_TRAIN else None)
    env = FireResetEnv(env)
    env = MaxAndSkipEnv(env)
    env = ClipRewardEnv(env)
    return env
env = reset_train_env()

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.cpu:
    DEVICE = torch.device("cpu")

# Hyperparameters
LR = args.lr
BATCH_SIZE = args.batch_size
BUFFER_SIZE = args.buffer_size
GAMMA = args.gamma
EPS_START = args.epsilon_start
EPS_END = args.epsilon_end
EPS_DECAY = args.epsilon_decay
N_STEPS = args.steps
MAX_STEPS = args.max_steps if args.max_steps else env.spec.max_episode_steps
args.max_steps = MAX_STEPS
WARM_UP = args.warm_up
TEST_EPS = 10
TEST_EVERY = 10_000

# Save options
SAVE_MODEL = args.save_model
SAVE_AGENT = args.save_agent

# Print info
ArgPrinter.print_banner()
ArgPrinter.print_env(ENV)
ArgPrinter.print_device(str(DEVICE))
ArgPrinter.print_args(args)

# State / action dims
n_actions = env.action_space.n

# Build agent
model = DQN(n_actions).to(DEVICE)
optimizer = optim.RMSprop(model.parameters(), lr=LR)
loss_function = nn.MSELoss()
replay_buffer = SimpleFrameBuffer(BUFFER_SIZE, DEVICE)
policy = EpsilonGreedyPolicy(model, DEVICE, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY)

agent = DQNAgent(model,
                 replay_buffer,
                 policy,
                 optimizer,
                 loss_function,
                 gamma=GAMMA,
                 device=DEVICE,
                )

# Train
print_busy("Train...")
train_scores = []

# Stop between training steps to record reward/maxQ
for _ in range(N_STEPS // TEST_EVERY):
    env = reset_train_env()
    agent.train(env,
                TEST_EVERY,
                MAX_STEPS,
                batch_size=BATCH_SIZE,
                warm_up_period=WARM_UP,
                )
    env = reset_train_env()
    scores = agent.play(env, TEST_EPS, env.spec.max_episode_steps)
    train_scores.append(np.mean(scores))

# Save training scores
try:
    score_dir = Path("../scores")
    if not score_dir.is_dir():
        score_dir.mkdir()
        print_success(f"Created folder:\t'{score_dir}'")

    idx = len([c for c in score_dir.iterdir()])
    np.save(score_dir / f"{ENV}_scores_{idx}.npy", train_scores)
    print_success("Saved rewards")
    
except Exception as e:
    print_fail("Could not save scores:")
    print(e)

print_success("Done")

# Free GPU memory
# TODO: think this could be done more efficiently
replay_buffer.to(torch.device("cpu"))
torch.cuda.empty_cache()

plot_scores(train_scores, title=(ENV + agent.name + " Training"))

# Reset env
# TODO: test in non-deterministic env?
env = gym.make(ENV_NAMES[ENV], render_mode="human" if VIS_EVAL else None)
env = FireResetEnv(env)
env = MaxAndSkipEnv(env)
env = ClipRewardEnv(env)

# Test
print_busy("Evaluate...")
threshold = env.spec.reward_threshold
if threshold:
    print_success(f"Target Score: {threshold:.2f}")
agent.load_best_model()
test_scores = agent.play(env, TEST_EPS, env.spec.max_episode_steps)

print_success("Done")
plot_scores(test_scores, title=(ENV + agent.name + " Test"))

# Save model
if SAVE_MODEL:
    try:
        model_dir = Path("../models") / ENV
        if not model_dir.is_dir():
            model_dir.mkdir(parents=True)
            print_success(f"Created folder:\t'{model_dir}'")

        idx = len([c for c in model_dir.iterdir()])
        model_file = f"{ENV}_{agent.name}_{idx}.pth"
        torch.save(model.state_dict(), model_dir / model_file)
        print_success("Saved model as:\t'" + model_file + "'")
        
    except Exception as e:
        print_fail("Could not save model:")
        print(e)

# Save agent
if SAVE_AGENT:
    try:
        agent_dir = Path("../agents") / ENV
        if not agent_dir.is_dir():
            agent_dir.mkdir(parents=True)
            print_success(f"Created folder:\t'{agent_dir}'")

        idx = len([c for c in agent_dir.iterdir()])
        agent_file = f"{ENV}_{agent.name}_{agent}_{idx}.pickle"
        with open(agent_dir / agent_file, "wb") as f:
            pickle.dump(agent, f)

        print_success("Save agent as:\t'" + agent_file + "'")

    except Exception as e:
        print_fail("Could not save agent:")
        print(e)
