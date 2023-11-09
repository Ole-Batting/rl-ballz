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
from torch_base import PolicyNetwork, ValueNetwork


@dataclass
class PPO:
    learning_rate: float = 0.001
    gamma: float = 0.9
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    policy_net: Optional[nn.Module] = None
    value_net: Optional[nn.Module] = None
    policy_optimizer: Optional[optim.Optimizer] = None
    value_optimizer: Optional[optim.Optimizer] = None
    n_channels_cnn: int = 4
    n_layers_cnn: int = 2
    n_layers_fcn: int = 2
    input_dim: Optional[int] = None
    env: Optional[object] = None
    curve: Optional[list] = None
    logger: Any = None
    batch_size: int = 1

    def setup(self, env, logger):
        self.env = env
        self.logger = logger
        self.curve = list()
        state = self.env.state
        ch_in, height, width = state.shape
        self.policy_net = PolicyNetwork(ch_in, self.n_channels_cnn, width, height, self.n_layers_cnn, self.n_layers_fcn, SPLITS)
        self.value_net = ValueNetwork(ch_in, self.n_channels_cnn, width, height, self.n_layers_cnn, self.n_layers_fcn)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        print(self.policy_net)
        print(self.value_net)

    def train(self, epochs=4096):
        # Training loop
        for epoch in tqdm(range(epochs)):
            # Collect data for policy update
            states = []
            actions = []
            rewards = []
            old_action_probs = []
            values = []

            done = False
            key = 0
            state = self.env.reset()
            
            while not done and key != ord("q"):
                state = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32)
                action_prob = self.policy_net(state)[0]
                value = self.value_net(state)[0]
                
                action = torch.multinomial(action_prob, 1).item()
                new_state, reward, done, key = self.env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                old_action_probs.append(action_prob[action])
                values.append(value)
                
                state = new_state
            
            self.curve.append(sum(rewards))
            self.logger.log({
                "reward-sum": sum(rewards),
                "avg10-reward-sum": np.mean(self.curve[-10:]),
                "min10-reward-sum": np.min(self.curve[-10:]),
                "rounds-cleared": self.env.i,
                "epoch": epoch,
            })
            if self.env.show_this:
                self.logger.log({
                    "video": wandb.Video(np.array(self.env.video_frames).transpose(0,3,1,2)[:,::-1], fps=60, format="mp4")
                })

            if key == ord("q"):
                print("quitting!")
                break
            
            # Calculate advantages
            returns = []
            advantages = []
            R = 0
            
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            
            for i in range(len(returns)):
                advantages.append(returns[i] - values[i])
            
            advantages = torch.tensor(advantages, dtype=torch.float32)
            returns = torch.tensor(returns, dtype=torch.float32)
            old_action_probs = torch.stack(old_action_probs)

            # Perform policy optimization
            for _ in range(self.batch_size):
                action_prob = self.policy_net(torch.cat(states, dim=0))
                new_action_probs = action_prob.gather(1, torch.tensor(actions).unsqueeze(1))
                
                ratio = new_action_probs / old_action_probs.detach()
                surrogate_obj = torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages)
                policy_loss = -torch.mean(surrogate_obj)
                
                # Value function update
                value = self.value_net(torch.cat(states, dim=0))
                value_loss = nn.MSELoss()(value.squeeze(), returns)
                
                # Entropy regularization
                entropy = -torch.sum(action_prob * torch.log(action_prob + 1e-10), dim=-1)
                entropy_loss = -torch.mean(entropy)
                
                # Total loss
                loss = policy_loss + self.value_coefficient * value_loss + self.entropy_coefficient * entropy_loss
                
                # Update networks
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()

    def close(self):
        # Close the environment
        self.env.close()