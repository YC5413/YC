#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Q-Network (DQN) 實作：Atari Boxing (Boxing-v5)
使用 PyTorch 原生實現，並支援：
  • Checkpoint 儲存／載入，方便「斷點續訓」
  • 環境預處理（灰度化、跳幀、裁剪、FrameStack）
  • Replay Buffer 經驗回放機制
  • ε-貪心策略選擇動作
  • Online / Target Network 同步更新（固定 target network）
  • Loss 計算、反向傳播、參數更新
  • 定期儲存模型，並續訓

主要流程：
 1. 環境建立與預處理
 2. DQN 網路與優化器初始化
 3. Checkpoint 檢查與載入
 4. 主訓練迴圈：收集經驗、緩衝、批次更新
 5. 定期同步 Target 網路（減少訓練不穩定）
 6. 定期儲存模型參數
"""

import os
import random
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStack

# === 超參數配置 =================================================
ENV_ID        = "ALE/Boxing-v5"        # Atari Boxing 環境識別碼
SEED          = 42                      # 隨機種子，確保結果可重現
BUFFER_SIZE   = 100_000                 # Replay Buffer 最大容量
BATCH_SIZE    = 32                      # 批次訓練大小
GAMMA         = 0.99                    # 折扣因子 γ，控制未來獎勵影響
LR            = 1e-4                    # Adam 優化器學習率
TARGET_UPDATE = 1000                    # 每隔多少步同步一次 Target 網路
INITIAL_EXP   = 10_000                  # 預先收集多少經驗再開始學習
MAX_STEPS     = 500_000                 # 訓練總步數上限
EPS_START     = 1.0                     # ε-貪心策略初始 ε 值（完全隨機）
EPS_END       = 0.1                     # ε-貪心策略最小 ε 值（保留探索）
EPS_DECAY     = 200_000                 # ε 線性衰減的總步數

# === Checkpoint 儲存設定 ========================================
MODEL_DIR     = "models"              # 模型與檔案儲存資料夾
MODEL_FILE    = os.path.join(MODEL_DIR, "dqn_boxing.pth")

# === Replay Buffer 定義 ==========================================
# Transition: 單步經驗五元組，用於存儲與回放
Transition = namedtuple("Transition",
                        ("state",      # 當前狀態
                         "action",     # 在當前狀態下採取的動作
                         "reward",     # 執行動作後獲得的獎勵
                         "next_state", # 下個狀態
                         "done"))      # 該步是否結束 Episode

class ReplayBuffer:
    """
    循環緩衝區，用 deque 自動丟棄最舊經驗
    提供 push(), sample(), __len__() 接口
    """
    def __init__(self, capacity):
        # maxlen=capacity 會自動在新增時丟棄最舊元素
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """將一筆 Transition 加入緩衝區"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        """
        隨機抽樣 batch_size 筆
        回傳 Transition of lists，方便後續打包成 tensor
        """
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        """目前緩衝區大小"""
        return len(self.buffer)

# === Q-Network 定義 (CNN + FC) ===================================
class QNetwork(nn.Module):
    """
    DQN 網路架構：
      - 三層 Conv2d + ReLU 提取影像特徵
      - 展平後兩層 Linear + ReLU，最後一層輸出動作 Q 值
    輸入：shape=(batch, in_channels, 84,84)，uint8 range [0,255]
    輸出：shape=(batch, n_actions)
    """
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        # 建立卷積層序列
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4),  # kernel=8, stride=4
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),           # kernel=4, stride=2
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),           # kernel=3, stride=1
            nn.ReLU()
        )
        # 動態計算展平特徵數量 n_flatten
        with torch.no_grad():
            dummy     = torch.zeros(1, in_channels, 84, 84)
            n_flatten = self.conv(dummy).view(1, -1).shape[1]
        # 建立全連接層序列
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        """
        前向傳播：
          1. uint8 -> float [0,1]
          2. 經過卷積、展平
          3. 經過全連接層輸出 Q 值
        """
        x = x / 255.0              # 歸一化至 [0,1]
        x = self.conv(x)          # 卷積特徵提取
        x = x.view(x.size(0), -1) # 展平 (batch, n_flatten)
        return self.fc(x)         # 輸出 (batch, n_actions)

# === 訓練主流程 ====================================================
def train():
    # 1. 建立並預處理環境
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = AtariPreprocessing(env,
                              frame_skip=4,
                              grayscale_obs=True,
                              scale_obs=False)
    env = FrameStack(env, num_stack=4)  # 堆疊 4 幀, shape->(4,84,84)

    # 2. 設置隨機種子，確保結果可重現
    env.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # 3. 選擇運算裝置 (MPS for Apple GPU / CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Info] Training on device: {device}")

    # 4. 初始化 online & target 網路 & optimizer & replay buffer
    n_actions  = env.action_space.n
    online_net = QNetwork(4, n_actions).to(device)
    target_net = QNetwork(4, n_actions).to(device)
    optimizer  = optim.Adam(online_net.parameters(), lr=LR)
    replay_buf = ReplayBuffer(BUFFER_SIZE)

    # 5. Checkpoint: 載入已儲存模型，若無則初始化 target=online
    start_step = 1
    if os.path.exists(MODEL_FILE):
        print("[Info] 載入 checkpoint，繼續訓練...")
        ckpt = torch.load(MODEL_FILE, map_location=device)
        online_net.load_state_dict(ckpt["online_state_dict"])
        target_net.load_state_dict(ckpt["target_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", INITIAL_EXP)
    else:
        os.makedirs(MODEL_DIR, exist_ok=True)
        target_net.load_state_dict(online_net.state_dict())

    # 6. 環境重置，初始化狀態與參數
    state, _ = env.reset()
    state    = np.array(state)  # shape=(4,84,84)
    episode_reward = 0
    epsilon        = EPS_START

    # 7. 主訓練迴圈: 互動、收集經驗、訓練網路
    for step in range(start_step, MAX_STEPS+1):
        # 7.1 ε-貪心策略選動作
        if random.random() < epsilon:
            action = env.action_space.sample()  # 隨機探索
        else:
            with torch.no_grad():
                # 將單筆 state 轉成 tensor, shape=(1,4,84,84)
                s_tensor = torch.tensor([state],
                                        dtype=torch.float32,
                                        device=device)
                # forward 計算 Q 值，選擇最大者
                action = int(online_net(s_tensor).argmax(1))

        # 7.2 與環境執行一步
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.array(next_state)
        episode_reward += reward

        # 7.3 存入 replay buffer
        replay_buf.push(state, action, reward, next_state, done)
        state = next_state

        # 7.4 Episode 結束: 重置環境並顯示獎勵
        if done:
            state, _ = env.reset()
            state    = np.array(state)
            print(f"[Episode] Step {step}, Reward: {episode_reward:.1f}, ε={epsilon:.3f}")
            episode_reward = 0

        # 7.5 線性衰減 ε，保證最低 EPS_END
        epsilon = max(EPS_END, EPS_START - step / EPS_DECAY)

        # 7.6 經驗累積足夠後開始更新網路
        if step > INITIAL_EXP and len(replay_buf) >= BATCH_SIZE:
            # 7.6.1 抽樣一個批次
            batch = replay_buf.sample(BATCH_SIZE)
            states      = torch.tensor(np.array(batch.state),
                                       dtype=torch.float32,
                                       device=device)
            actions     = torch.tensor(batch.action,
                                       dtype=torch.long,
                                       device=device).unsqueeze(1)
            rewards     = torch.tensor(batch.reward,
                                       dtype=torch.float32,
                                       device=device).unsqueeze(1)
            next_states = torch.tensor(np.array(batch.next_state),
                                       dtype=torch.float32,
                                       device=device)
            dones       = torch.tensor(batch.done,
                                       dtype=torch.float32,
                                       device=device).unsqueeze(1)

            # 7.6.2 計算 target Q 值: r + γ * max_a Q_target(next_state, a) * (1-done)
            with torch.no_grad():
                q_next   = target_net(next_states).max(1, keepdim=True)[0]
                q_target = rewards + GAMMA * q_next * (1 - dones)

            # 7.6.3 計算 online Q_current
            q_current = online_net(states).gather(1, actions)

            # 7.6.4 計算 MSE loss, 反向傳播, 更新參數
            loss = nn.MSELoss()(q_current, q_target)
            optimizer.zero_grad()  # 將所有參數梯度歸零
            loss.backward()        # 自動微分: 計算梯度
            optimizer.step()       # 參數更新

            # 7.6.5 同步 Target 網路
            if step % TARGET_UPDATE == 0:
                target_net.load_state_dict(online_net.state_dict())

        # 8. 定期儲存 Checkpoint
        if step % TARGET_UPDATE == 0:
            torch.save({
                "step": step + 1,
                "online_state_dict":    online_net.state_dict(),
                "target_state_dict":    target_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, MODEL_FILE)
            print(f"[Checkpoint] 已儲存至 {MODEL_FILE}，下次從 Step {step+1} 繼續")

    # 9. 結束訓練，關閉環境
    env.close()
    print("訓練完成。")

# === 主程式入口 ====================================================
if __name__ == "__main__":
    train()