#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


def make_env():
    """
    建立單一個 Boxing 環境（gymnasium 格式）。
    這裡使用 ID "ALE/Boxing-v5"（gymnasium 推薦新版），
    並以 render_mode="rgb_array" 來取得畫面陣列，可以後續用 .render() 顯示。
    """
    env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")
    return env


if __name__ == "__main__":
    # 用 DummyVecEnv 將單一環境包成 vectorized 環境
    env = DummyVecEnv([make_env])

    # VecTransposeImage 會把 [H, W, C] → [C, H, W]，符合 PyTorch CNN 輸入
    env = VecTransposeImage(env)

    # 選擇裝置：如果 M1/M2/M3 可用 mps，就用 mps，否則 fallback 到 cpu
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 建立 PPO 模型；normalize_images=True 是預設，會自動把 uint8 → float32 / 255
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=device,
        # 如果想關閉 SB3 的自動 normalize，可加以下參數（但一般不需要）：
        # policy_kwargs={"normalize_images": False}
    )

    # 訓練 100k 步（可以自行改成更大的數字）
    TIMESTEPS = 100_000
    model.learn(total_timesteps=TIMESTEPS)

    # 測試：reset() 會回傳 (obs, info)，這裡直接 unpack
    obs, info = env.reset()
    for _ in range(5_000):
        # 顯示畫面
        env.render()
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # 如果一集結束，就重新 reset
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("訓練與測試完成！")
