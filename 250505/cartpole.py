# 導入必要的庫
import gymnasium as gym  # 強化學習環境庫
import torch  # 深度學習框架
import torch.nn as nn  # 神經網絡模塊
import torch.optim as optim  # 優化器
import random  # 用於隨機選擇動作
import numpy as np  # 數值計算庫
from collections import deque  # 用於實現經驗回放緩衝區

class DQN(nn.Module):  # 定義深度Q網絡模型
    def __init__(self, state_dim, action_dim):  # 初始化函數
        super(DQN, self).__init__()
        self.fc = nn.Sequential(  # 定義全連接神經網絡
            nn.Linear(state_dim, 128),  # 輸入層到隱藏層
            nn.ReLU(),  # 激活函數
            nn.Linear(128, action_dim)  # 隱藏層到輸出層
        )

    def forward(self, x):  # 前向傳播函數
        return self.fc(x)  # 返回預測的Q值

# 創建遊戲環境
env = gym.make('CartPole-v1', render_mode='human')  # 創建CartPole環境並啟用視覺化
state_dim = env.observation_space.shape[0]  # 狀態空間維度（4個特徵：位置、速度、角度、角速度）
action_dim = env.action_space.n  # 動作空間維度（2個動作：向左或向右推）

# 創建神經網絡
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

# 設置優化器和損失函數
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)  # Adam優化器，學習率為0.001
loss_fn = nn.MSELoss()  # 均方誤差損失函數

# 設置經驗回放緩衝區
memory_capacity = 10000  # 記憶體容量
memory = deque(maxlen=memory_capacity)  # 使用雙端隊列存儲經驗
batch_size = 64  # 批次大小

# 設置強化學習的超參數
gamma = 0.99  # 折扣因子，用於計算未來獎勵的現值
epsilon = 1.0  # 初始探索率
epsilon_min = 0.01  # 最小探索率
epsilon_decay = 0.995  # 探索率衰減係數
episodes = 300  # 訓練回合數
target_update = 10  # 目標網絡更新頻率

# 訓練統計
best_reward = 0  # 記錄最佳得分
episode_rewards = []  # 存儲每個回合的得分

# 開始訓練循環
for episode in range(episodes):  # 遍歷每個訓練回合
    state, _ = env.reset()  # 重置環境，獲取初始狀態
    state = torch.FloatTensor(state)  # 將狀態轉換為張量
    total_reward = 0  # 當前回合的總獎勵
    done = False  # 回合結束標誌
    steps = 0  # 當前回合的步數

    while not done:  # 當回合未結束時
        # 選擇動作：探索或利用
        if random.random() < epsilon:  # epsilon概率選擇隨機動作（探索）
            action = random.randint(0, action_dim - 1)
        else:  # 選擇Q值最大的動作（利用）
            with torch.no_grad():  # 不計算梯度
                action = policy_net(state).argmax().item()

        # 執行動作並獲得反饋
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # 判斷是否結束
        next_state = torch.FloatTensor(next_state)  # 將下一狀態轉換為張量

        # 存儲經驗到記憶體
        memory.append((state, action, reward, next_state, done))
        state = next_state  # 更新當前狀態
        total_reward += reward  # 累加獎勵
        steps += 1  # 增加步數

        # 當記憶體中有足夠的樣本時進行訓練
        if len(memory) >= batch_size:
            # 從記憶體中隨機採樣
            batch = random.sample(list(memory), batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # 將數據轉換為張量
            states = torch.stack(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.stack(next_states)
            dones = torch.BoolTensor(dones).unsqueeze(1)

            # 計算當前Q值和目標Q值
            q_values = policy_net(states).gather(1, actions)  # 當前動作的Q值
            max_next_q = target_net(next_states).max(1)[0].detach().unsqueeze(1)  # 下一狀態的最大Q值
            target_q = rewards + gamma * max_next_q * (~dones)  # 計算目標Q值

            # 計算損失並更新網絡
            loss = loss_fn(q_values, target_q)  # 計算損失
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向傳播
            optimizer.step()  # 更新參數

    # 更新探索率
    if epsilon > epsilon_min:  # 如果探索率大於最小值
        epsilon *= epsilon_decay  # 衰減探索率

    # 定期更新目標網絡
    if (episode + 1) % target_update == 0:  # 每target_update個回合
        target_net.load_state_dict(policy_net.state_dict())  # 更新目標網絡

    # 更新並記錄訓練數據
    if total_reward > best_reward:  # 如果獲得了更好的得分
        best_reward = total_reward  # 更新最佳得分

    episode_rewards.append(total_reward)  # 記錄當前回合的得分
    avg_reward = np.mean(episode_rewards[-100:])  # 計算最近100回合的平均得分

    # 打印訓練信息
    print(f"Episode {episode+1}")  # 當前回合數
    print(f"Total Steps: {steps}")  # 本回合的總步數
    print(f"Total Reward: {total_reward:.1f}")  # 本回合的總獎勵
    print(f"Best Reward: {best_reward:.1f}")  # 最佳獎勵記錄
    print(f"Average Reward (last 100): {avg_reward:.1f}")  # 最近100回合的平均獎勵
    print(f"Epsilon: {epsilon:.3f}")  # 當前探索率
    print("-" * 50)  # 分隔線

# 關閉環境
env.close()  # 程序結束時關閉環境
