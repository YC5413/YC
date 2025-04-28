import gymnasium as gym
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import time

# 環境初始化
env = gym.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Q-table 初始化
Q = np.zeros((n_states, n_actions))

# 超參數
learning_rate = 0.8
gamma = 0.95
episodes = 200

# 探索率與 decay 設定
initial_epsilon = 0.3
min_epsilon = 0.01
epsilon_decay = 0.995
epsilon = initial_epsilon

def plot_maze(state, env, Q, episode=None, epsilon=None, success_rate=None, nrow=4, ncol=4):
    desc = env.unwrapped.desc.astype(str)
    maze = np.zeros((nrow, ncol, 3), dtype=np.uint8) + 255
    
    # 顏色設定
    for r in range(nrow):
        for c in range(ncol):
            if desc[r, c] == 'H':
                maze[r, c] = [0, 0, 0]
            elif desc[r, c] == 'G':
                maze[r, c] = [255, 215, 0]
            elif desc[r, c] == 'S':
                maze[r, c] = [173, 216, 230]
    
    agent_r, agent_c = state // ncol, state % ncol
    maze[agent_r, agent_c] = [255, 0, 0]
    
    plt.clf()
    
    if episode is not None:
        plt.subplot(2, 1, 1)
        plt.axis('off')
        info_text = f'回合: {episode + 1}/{episodes}\n'
        info_text += f'探索率: {epsilon:.3f}\n'
        if success_rate is not None:
            info_text += f'成功率: {success_rate:.1%}'
        plt.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10)
        plt.subplot(2, 1, 2)
    
    plt.imshow(maze)
    plt.title('FrozenLake 迷宮進度 (顯示各方向Q值)')
    plt.axis('off')
    
    directions = ['↑', '→', '↓', '←']
    for r in range(nrow):
        for c in range(ncol):
            s = r * ncol + c
            q_values = [f"{directions[i]}{Q[s,i]:.1f}" for i in range(Q.shape[1])]
            q_str = f"{q_values[0]}\n{q_values[3]} {q_values[1]}\n{q_values[2]}"
            color = 'white' if np.sum(maze[r, c]) < 400 else 'black'
            plt.text(c, r, q_str, 
                    ha='center', va='center', fontsize=7,
                    color=color, fontweight='bold',
                    bbox=dict(facecolor='none', edgecolor='none', alpha=0.7, pad=0))
    plt.pause(0.01)

plt.ion()
fig = plt.figure(figsize=(6, 8))

# 訓練過程
success_count = 0
for episode in range(episodes):
    state, _ = env.reset()
    state = int(state)
    step = 0
    done = False
    reached_goal = False
    qtable_before = Q.copy()
    
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_state, env_reward, terminated, truncated, info = env.step(action)
        next_state = int(next_state)
        done = terminated or truncated
        
        current_row = state // 4
        current_col = state % 4
        next_row = next_state // 4
        next_col = next_state % 4
        
        if current_row == next_row and current_col == next_col:
            reward = -5
        elif done and env_reward == 1:
            reward = 100
            reached_goal = True
        elif done and env_reward == 0:
            reward = -10
        else:
            reward = -0.1

        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state, best_next_action]
        td_error = td_target - Q[state, action]
        Q[state, action] += learning_rate * td_error
        
        if reached_goal:
            success_count += 1
        success_rate = success_count / (episode + 1)
        plot_maze(state, env, Q, episode, epsilon, success_rate)
        
        state = next_state
        step += 1  
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
        epsilon = max(epsilon, min_epsilon)
        
    q_change = np.sum(np.abs(Q - qtable_before))
    if reached_goal:
        print(f"[心得] 第 {episode+1} 回合：成功到達終點！Q-Table 變化量：{q_change:.2f}")
    else:
        print(f"[心得] 第 {episode+1} 回合：有學習新路徑。Q-Table 變化量：{q_change:.2f}")

plt.ioff()
# 測試階段
state, _ = env.reset()
state = int(state)
done = False

while not done:
    action = np.argmax(Q[state])
    next_state, reward, terminated, truncated, info = env.step(action)
    next_state = int(next_state)
    done = terminated or truncated
    plot_maze(next_state, env, Q)  # 測試階段不顯示訓練資訊
    time.sleep(0.05)
    state = next_state

plt.show()


