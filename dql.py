from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from constants import W, H, SPLITS
from torch_base import DeepQNetwork

@dataclass
class DQL:
    learning_rate: float = 0.01
    gamma: float = 0.9
    epsilon: float = 0.2
    q_net: Optional[nn.Module] = None
    optimizer: Optional[optim.Optimizer] = None
    n_channels_cnn: int = 4
    n_layers_cnn: int = 2
    n_layers_fcn: int = 2
    input_dim: Optional[int] = None
    env: Optional[object] = None
    curve: Optional[list] = None
    logger: Any = None

    def setup(self, env, logger):
        self.env = env
        self.logger = logger
        self.curve = list()
        state = self.env.state
        ch_in, height, width = state.shape
        self.q_net = DeepQNetwork(ch_in, self.n_channels_cnn, width, height, self.n_layers_cnn, self.n_layers_fcn, SPLITS)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        print(self.q_net)

    def train(self, num_episodes=1001):
        for episode in tqdm(range(num_episodes)):
            rewards = []

            state = torch.tensor(np.expand_dims(self.env.state, axis=0), dtype=torch.float32)
            done = False
            key = 0

            while not done and key != ord("q"):
                q_values = self.q_net(state)[0]
                if np.random.rand() < self.epsilon:
                    action = torch.multinomial(nn.functional.softmax(q_values.clone(), dim=0), 1).item()
                else:
                    action = torch.argmax(q_values).item()

                next_state, reward, done, key = self.env.step(action)
                next_state = torch.tensor(np.expand_dims(next_state, axis=0), dtype=torch.float32)
                max_q_value = torch.max(self.q_net(next_state)[0])
                target_q_value = reward + self.gamma * max_q_value

                loss = nn.MSELoss()(q_values[action], target_q_value)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                rewards.append(reward)

                state = next_state

                self.logger.log(dict(loss=loss.item(), round=self.env.i))
            
            self.curve.append(sum(rewards))
            self.logger.log({
                "reward-sum": sum(rewards),
                "rounds-cleared": self.env.i - 8,
                "episode": episode,
            })
            if self.env.show_this:
                self.logger.log({
                    "video": wandb.Video(np.array(self.env.video_frames).transpose(0,3,1,2)[:,::-1], fps=60, format="mp4")
                })

            self.env.reset()

    def close(self):
        # Close the environment
        self.env.close()