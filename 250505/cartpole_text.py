import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(DQN,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,action_dim)
        )
    def forward(self,x):
        return self.fc(x)
        
env = gym.make('CartPole-v1',render_mode = 'human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim,action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(),lr=0.001)
loss_fn = nn.MSELoss()

memory_capacity = 10000
memory = deque(maxlen=memory_capacity)
batch_size = 64

gamma = 0.99
epsilon = 1.0  # 初始探索率
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 300
target_update = 10

best_reward = 0
episode_rewards = []

for episode in range(episodes):
    state,_ = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0
    done = False
    steps = 0
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state).argmax().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        steps += 1
        if len(memory) >= batch_size:
            batch = random.sample(list(memory), batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.stack(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.stack(next_states)
            dones = torch.BoolTensor(dones).unsqueeze(1)
            q_values = policy_net(states).gather(1, actions)
            max_next_q = target_net(next_states).max(1)[0].detach().unsqueeze(1)
            target_q = rewards + gamma * max_next_q * (~dones)
            loss = loss_fn(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    if (episode + 1) % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if total_reward > best_reward:
        best_reward = total_reward
    episode_rewards.append(total_reward)
    avg_reward = np.mean(episode_rewards[-100:])
    print(f"Episode {episode+1}")
    print(f"Total Steps: {steps}")
    print(f"Total Reward: {total_reward:.1f}")
    print(f"Best Reward: {best_reward:.1f}")
    print(f"Average Reward (last 100): {avg_reward:.1f}")
    print(f"Epsilon: {epsilon:.3f}")
    print("-" * 50)
env.close()