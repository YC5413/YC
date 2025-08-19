#!/usr/bin/env python3
import torch
import torch.nn as nn
import random
import numpy as np
import collections
import pickle
import os

# ---------------- Dueling DQN Network ----------------
class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out = self._get_conv_out(input_shape)
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        features = self.conv(x).view(x.size(0), -1)
        value = self.fc_value(features)
        advantage = self.fc_advantage(features)
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals

# ---------------- Double Dueling DQN Agent ----------------
class DQNAgent:
    def __init__(self, state_shape, n_actions, lr=1e-4, gamma=0.99, batch_size=32, model_path=None, stage_epsilons_path=None, replay_buffer_path=None, max_memory_size=100000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.batch_size = batch_size
        self.stage_epsilons_path = stage_epsilons_path  
        self.replay_buffer_path = replay_buffer_path
        self.max_memory_size = max_memory_size

        self.online_net = DuelingDQNNetwork(state_shape, n_actions).to(self.device)
        self.target_net = DuelingDQNNetwork(state_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = collections.deque(maxlen=max_memory_size)

        self.steps_done = 0
        self.target_update_freq = 1000


        self.stage_epsilons = {}
        self.visited_stages = set()  
        self.current_stage_visits = {}

        if model_path:
            self.load(model_path)
            try:
                eps_path = self.stage_epsilons_path if self.stage_epsilons_path else "stage_epsilons.pkl"
                with open(eps_path, 'rb') as f:
                    self.stage_epsilons = pickle.load(f)
                print(f"Loaded stage epsilons: {self.stage_epsilons}")
            except (FileNotFoundError, EOFError):
                print("No stage epsilons file found or file is empty. Starting with fresh epsilons.")
                

            if self.replay_buffer_path and os.path.exists(self.replay_buffer_path):
                self.load_replay_buffer()

    def track_stage_visit(self, stage):
        self.visited_stages.add(stage)
        if stage in self.current_stage_visits:
            self.current_stage_visits[stage] += 1
        else:
            self.current_stage_visits[stage] = 1

    def get_stage_epsilon(self, stage, default_epsilon=0.01):
        return self.stage_epsilons.get(stage, default_epsilon)

    def set_stage_epsilon(self, stage, value):
        self.stage_epsilons[stage] = value

    def decay_stage_epsilon(self, stage, min_epsilon, decay_factor):
        if stage in self.stage_epsilons:
            old_eps = self.stage_epsilons[stage]
            new_eps = max(min_epsilon, old_eps * decay_factor)
            self.stage_epsilons[stage] = new_eps
            return new_eps
        return min_epsilon

    def select_action(self, state, epsilon=0.01, stage=None):
        if stage is not None and stage in self.stage_epsilons:
            epsilon = self.stage_epsilons[stage]
        if random.random() < epsilon:
            return random.randint(0, self.online_net.fc_advantage[-1].out_features - 1)
        with torch.no_grad():
            state = state.to(self.device).unsqueeze(0)
            q_values = self.online_net(state)
            return torch.argmax(q_values, dim=1).item()

    def memorize(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float).unsqueeze(1)

        current_q = self.online_net(states).gather(1, actions)

        next_actions = self.online_net(next_states).argmax(1, keepdim=True)
        next_q = self.target_net(next_states).gather(1, next_actions)

        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path):
        checkpoint = {
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stage_epsilons': self.stage_epsilons,
            'visited_stages': list(self.visited_stages),
            'current_stage_visits': self.current_stage_visits,
            'steps_done': self.steps_done
        }
        torch.save(checkpoint, path)
        eps_path = self.stage_epsilons_path if self.stage_epsilons_path else "stage_epsilons.pkl"
        with open(eps_path, "wb") as f:
            pickle.dump(self.stage_epsilons, f)

    def save_replay_buffer(self):
        if self.replay_buffer_path:
            try:
                with open(self.replay_buffer_path, 'wb') as f:
                    pickle.dump(list(self.replay_buffer), f)
                print(f"Saved replay buffer to {self.replay_buffer_path} (size: {len(self.replay_buffer)})")
            except Exception as e:
                print(f"Error saving replay buffer: {e}")
    
    def load_replay_buffer(self):
        if self.replay_buffer_path and os.path.exists(self.replay_buffer_path):
            try:
                print(f"嘗試從 {self.replay_buffer_path} 載入replay buffer...")
                with open(self.replay_buffer_path, 'rb') as f:
                    loaded_buffer = pickle.load(f)
                    self.replay_buffer = collections.deque(loaded_buffer, maxlen=self.replay_buffer.maxlen)
                print(f"Loaded replay buffer from {self.replay_buffer_path} (size: {len(self.replay_buffer)})")
            except (Exception, KeyboardInterrupt) as e:
                print(f"Error loading replay buffer: {e}")
                print("Creating new replay buffer")
                self.replay_buffer = collections.deque(maxlen=self.replay_buffer.maxlen)
        else:
            print("No replay buffer file found. Starting with fresh buffer.")
            self.replay_buffer = collections.deque(maxlen=self.replay_buffer.maxlen)
            
    def reset_memory(self, partial=False, keep_ratio=0.0):
        """重置記憶回放緩衝區，可以選擇部分重置或完全重置
        
        Args:
            partial (bool): 是否部分重置，如果為True，保留一部分記憶
            keep_ratio (float): 保留的記憶比例，介於0-1之間
        """
        if partial and keep_ratio > 0:
            # 保留一部分最新的記憶
            buffer_size = len(self.replay_buffer)
            keep_size = int(buffer_size * keep_ratio)
            new_buffer = collections.deque(list(self.replay_buffer)[-keep_size:], maxlen=self.replay_buffer.maxlen)
            self.replay_buffer = new_buffer
            print(f"部分重置記憶回放緩衝區: 保留了 {keep_size}/{buffer_size} ({keep_ratio*100:.1f}%) 的記憶")
        else:
            # 完全重置
            old_size = len(self.replay_buffer)
            self.replay_buffer = collections.deque(maxlen=self.replay_buffer.maxlen)
            print(f"完全重置記憶回放緩衝區: 刪除了 {old_size} 的記憶")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded optimizer state from checkpoint")
        
        if 'steps_done' in checkpoint:
            self.steps_done = checkpoint['steps_done']
            print(f"Loaded training steps: {self.steps_done}")
            
        if 'stage_epsilons' in checkpoint:
            self.stage_epsilons = checkpoint['stage_epsilons']
        if 'visited_stages' in checkpoint:
            self.visited_stages = set(checkpoint['visited_stages'])
        if 'current_stage_visits' in checkpoint:
            self.current_stage_visits = checkpoint['current_stage_visits']

    def __len__(self):
        return len(self.replay_buffer)
