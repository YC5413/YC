#!/usr/bin/env python3

import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from gymnasium.wrappers import RecordVideo

def make_env():
    """
    创建一个 Boxing-v5 环境实例，并用 RecordVideo 录制视频。
    video_folder：指定存放 .mp4 的目录
    episode_trigger=lambda e: True 表示录制所有 episode
    """
    env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")
    # 确保 “videos” 目录存在
    os.makedirs("videos", exist_ok=True)
    env = RecordVideo(
        env,
        video_folder="videos",
        name_prefix="boxing_run",
        episode_trigger=lambda episode_id: True,
    )
    return env

if __name__ == "__main__":
    # 1. 使用 DummyVecEnv 将单一环境包装为 vectorized 环境
    env = DummyVecEnv([make_env])

    # 2. VecTransposeImage 会把 [H, W, C] → [C, H, W] 并归一化为 float32/255
    env = VecTransposeImage(env)

    # 3. 选择训练设备：如果 Apple Silicon 的 mps 可用，优先使用 mps；否则使用 cpu
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 4. 初始化 PPO 模型
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=device,
        policy_kwargs={"normalize_images": True},
    )

    # 5. 开始训练，100k 步
    TIMESTEPS = 100_000
    model.learn(total_timesteps=TIMESTEPS)

    # 6. 训练完成后进行测试并继续录制
    obs, info = env.reset()
    for _ in range(5_000):
        env.render()  # 渲染游戏画面；也会录到 “videos” 文件夹
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    # 7. 关闭环境
    env.close()
    print("训练与测试完成！视频文件保存在 “videos/” 目录下。")
