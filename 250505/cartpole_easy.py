import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# 定義 DQN 模型
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# 建立環境
env = gym.make('CartPole-v1', render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 建立主網路與目標網路
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())  # 同步參數
target_net.eval()

# 優化器與損失函數
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Replay Buffer
memory = deque(maxlen=10000)
batch_size = 64

# 超參數
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 300
update_every = 10

# 訓練主迴圈
for episode in range(episodes):
    state, _ = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0

    done = False
    while not done:
        # ε-greedy 策略
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state).argmax().item()

        # 執行動作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state)

        # 儲存到 replay buffer
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # 若 buffer 足夠大，就訓練一次
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.stack(states)
            actions = torch.tensor(actions).unsqueeze(1)
            rewards = torch.tensor(rewards).unsqueeze(1)
            next_states = torch.stack(next_states)
            dones = torch.tensor(dones).unsqueeze(1)

            # 計算 Q 值與 TD target
            q_values = policy_net(states).gather(1, actions)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + gamma * max_next_q * (~dones)

            # 更新參數
            loss = loss_fn(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 更新 ε
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # 每 N 回合更新 target_net
    if (episode + 1) % update_every == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Ep {episode+1} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.3f}")

env.close()
