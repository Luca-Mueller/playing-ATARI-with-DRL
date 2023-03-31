import torch
from torch import nn
import torch.nn.functional as F


"""
    Deep Q-Network, original architecture
"""


class DQN(nn.Module):
    """
        Input:  frame stack of 84x84 frames
        Output: Q-values (call) / best action (predict)
    """
    def __init__(self, outputs: int, window_size: int = 4):
        super(DQN, self).__init__()
        h, w = 84, 84
        self.shape = ((window_size, h, w), outputs)
        self.conv1 = nn.Conv2d(window_size, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2)
        convh = conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2)
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.head = nn.Linear(256, outputs)

    def forward(self, x) -> torch.tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.head(x)

    def predict(self, x, device: str) -> torch.tensor:
        y = self.__call__(x).max(1)[1].view(1, 1)
        return torch.tensor([[y]], dtype=torch.long, device=device)
