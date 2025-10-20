# -*- coding: utf-8 -*-
import os
import time
from typing import Dict, Any
import gym

# 核心函式庫
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import QRDQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# 自定義環境包裝器 (用於灰階化與畫面縮放)
from gym.wrappers import GrayScaleObservation

import numpy as np
import cv2


class CropResize(gym.ObservationWrapper):
    """Crop the original frame and resize to 84x84 with a single channel.

    Cropping rectangle: obs[30:228, 8:248] -> shape (198, 240)
    Resize target: (84, 84)

    The wrapper returns dtype=np.uint8 and shape (84, 84, 1) to be compatible
    with SB3's image policies after VecTransposeImage and VecFrameStack.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 1),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # expect obs in HWC uint8
        # Crop
        cropped = obs[30:228, 8:248]
        # Resize using INTER_AREA for downsampling
        resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
        # Ensure single channel (if input was grayscale already, still keep shape)
        if resized.ndim == 2:
            out = resized.reshape(84, 84, 1)
        else:
            # If input still has 3 channels (RGB), convert to grayscale first
            if resized.shape[2] == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                out = gray.reshape(84, 84, 1)
            else:
                # Unexpected channel count, take first channel
                out = resized[:, :, 0].reshape(84, 84, 1)

        return out.astype(np.uint8)


class ResetAcceptSeed(gym.Wrapper):
    """Wrapper to make legacy Gym envs accept reset(seed=...) calls.

    Some older Gym envs don't accept a 'seed' keyword in reset(). SB3's
    DummyVecEnv may call env.reset(seed=...), which causes a TypeError.
    This wrapper swallows the seed kwarg and forwards to the underlying
    env.reset() call while still returning the observation.
    """
    def reset(self, *args, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        res = self.env.reset(*args, **kwargs)
        if not isinstance(res, tuple):
            return res, {}
        # tuple: prefer first two elements as (obs, info)
        if len(res) >= 2:
            return res[0], res[1]
        return res[0], {}


class StepToGymnasium(gym.Wrapper):
    """Normalize step() return to Gymnasium-style 5-tuple.

    Some older Gym envs return (obs, reward, done, info). SB3's
    DummyVecEnv (and shimmy) may expect the newer API with separate
    terminated and truncated flags. This wrapper converts 4-tuple
    returns into (obs, reward, terminated, truncated, info) with
    truncated=False when not provided.
    """
    def step(self, action, *args, **kwargs):
        res = self.env.step(action, *args, **kwargs)
        if isinstance(res, tuple) and len(res) == 5:
            return res
        if isinstance(res, tuple) and len(res) == 4:
            obs, reward, done, info = res
            return obs, reward, done, False, info
        return res


# Legacy Gym only: do not use env_compat wrappers

"""
================================================================================
瑪利歐強化學習實驗框架 (Requirement Specification & Implementation)
================================================================================
本腳本提供一個標準化的流程來訓練與評估不同的強化學習演算法在《超級瑪利歐兄弟》環境中的表現。

核心功能：
1.  **標準化環境建立**: 自動處理所有必要的環境預處理 (動作空間、灰階、縮放、幀堆疊)。
2.  **基準超參數管理**: 集中管理來自 RL Baselines Zoo 的 Atari 基準超參數，確保比較公平性。
3.  **模組化模型選擇**: 只需修改一個變數即可輕鬆切換 PPO, DQN, A2C, QRDQN 等不同演算法。
4.  **自動化評估與儲存**: 使用 Callback 在訓練過程中定期評估模型，並自動儲存表現最佳的模型。
5.  **清晰的日誌與模型管理**: 將訓練日誌 (Logs) 和模型 (Models) 儲存在以時間戳命名的資料夾中，方便追蹤。

遵循本框架，可以確保你的所有實驗都在一個受控且可比較的基準下進行。
"""

# --- 1. 環境建立函式 ---
def create_mario_env(env_id: str, num_envs: int):
    """Create a vectorized Mario environment with minimal, clear compatibility handling.

    Strategy:
    - Probe the raw env once to detect whether it uses Gymnasium-style reset/step
      semantics. Based on that, only apply the small, focused wrappers necessary.
    - Keep per-env wrappers (gray, resize) unchanged.
    """
    def make_env_fn(rank: int):
        def _init():
            env = gym_super_mario_bros.make(env_id)
            # defensive attributes for some shimmy/old envs
            if not hasattr(env, 'render_mode'):
                env.render_mode = None
            if not hasattr(env, 'metadata'):
                env.metadata = {}

            # apply JoypadSpace action mapping
            env = JoypadSpace(env, COMPLEX_MOVEMENT)

            # apply simple preprocessing wrappers: grayscale -> crop+resize
            env = GrayScaleObservation(env, keep_dim=True)
            env = CropResize(env)

            # Make legacy env accept reset(seed=...) calls
            env = ResetAcceptSeed(env)
            # Normalize step() to return (obs, reward, terminated, truncated, info)
            env = StepToGymnasium(env)

            # Legacy gym: no StepToGymnasium / ResetToGymnasium

            return env
        return _init

    env = DummyVecEnv([make_env_fn(i) for i in range(num_envs)])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    return env

# --- 2. 基準超參數字典 ---
# 這些參數主要參考自 Stable Baselines3 RL Baselines Zoo 的 Atari 遊戲設定
# 我們的目標不是找到絕對最佳參數，而是為所有演算法提供一個公平的、公認的基準點。
HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    'PPO': {
        'gamma': 0.99,
        'n_steps': 256,
        'batch_size': 256,
        'n_epochs': 4,
        'learning_rate': 2.5e-4,
        'clip_range': 0.1,
        'ent_coef': 0.01,
    },
    'DQN': {
        'gamma': 0.99,
        'learning_rate': 1e-4,
        'buffer_size': 100000,
        'learning_starts': 10000,
        'batch_size': 32,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.01,
    },
    'A2C': {
        'gamma': 0.99,
        'n_steps': 5,
        'learning_rate': 7e-4,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
    },
    'QRDQN': {
        'gamma': 0.99,
        'learning_rate': 1e-4,
        'buffer_size': 100000,
        'learning_starts': 10000,
        'batch_size': 32,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.01,
    },
}

# --- 3. 主執行區塊 ---
if __name__ == '__main__':
    
    # ==========================
    # ===== 實驗配置區 (請修改這裡或使用環境變數) =====
    # ==========================
    # Defaults — can be overridden by environment variables:
    #   MARIO_MODEL, MARIO_WORLD, MARIO_STAGE, MARIO_VERSION, N_ENVS, TOTAL_TIMESTEPS
    MODEL_CHOICE = os.environ.get('MARIO_MODEL', 'A2C')  # 可選: 'PPO', 'DQN', 'A2C', 'QRDQN'
    TOTAL_TIMESTEPS = int(os.environ.get('TOTAL_TIMESTEPS', '1000000'))
    N_ENVS = int(os.environ.get('N_ENVS', '4'))
    WORLD = int(os.environ.get('MARIO_WORLD', '1'))
    STAGE = int(os.environ.get('MARIO_STAGE', '1'))
    VERSION = os.environ.get('MARIO_VERSION', 'v0')
    ENV_ID = f"SuperMarioBros-{WORLD}-{STAGE}-{VERSION}"
    # ==========================
    
    # 建立唯一的實驗資料夾名稱
    log_time = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{MODEL_CHOICE}_{log_time}"
    
    # 設定儲存路徑
    log_dir = os.path.join("logs", experiment_name)
    model_dir = os.path.join("models", experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"===== 開始實驗: {experiment_name} =====")
    print(f"演算法: {MODEL_CHOICE}")
    print(f"總訓練步數: {TOTAL_TIMESTEPS}")
    print(f"日誌將儲存至: {log_dir}")
    print(f"模型將儲存至: {model_dir}")

    # 建立訓練環境
    print("正在建立環境...")
    env = create_mario_env(ENV_ID, N_ENVS)

    # 設定評估 Callback
    # EvalCallback 會在訓練期間定期在一個獨立的環境中評估模型，並儲存表現最好的模型
    eval_env = create_mario_env(ENV_ID, 1) # 評估通常用單一環境
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=os.path.join(model_dir, 'best_model'),
                                 log_path=log_dir,
                                 eval_freq=max(10000 // N_ENVS, 1), # 每 10000 步評估一次
                                 deterministic=True,
                                 render=False)

    # 根據選擇，實例化模型
    print(f"正在實例化模型: {MODEL_CHOICE}...")
    params = HYPERPARAMS[MODEL_CHOICE]
    # Device selection: default to CPU per user's request; allow optional override
    # via environment variable OPEN_MARIO_DEVICE (values: 'cpu' or 'cuda').
    env_device = os.environ.get('OPEN_MARIO_DEVICE', '').lower()
    if env_device == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    if MODEL_CHOICE == 'PPO':
        model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_dir, device=device, **params)
    elif MODEL_CHOICE == 'DQN':
        model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=log_dir, device=device, **params)
    elif MODEL_CHOICE == 'A2C':
        model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_dir, device=device, **params)
    elif MODEL_CHOICE == 'QRDQN':
        model = QRDQN('CnnPolicy', env, verbose=1, tensorboard_log=log_dir, device=device, **params)
    else:
        raise ValueError(f"未知的模型選擇: {MODEL_CHOICE}")
        
    # ** Dueling DQN 的設定 **
    # 如果你想使用 Dueling DQN, 只需要在 DQN 模型的 policy_kwargs 中加入 dueling=True 即可
    #範例: model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=log_dir, policy_kwargs={'dueling': True}, **params)

    # 開始訓練
    print("===== 模型開始訓練 =====")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    except KeyboardInterrupt:
        print("訓練被手動中斷。")
    finally:
        # 儲存最終模型
        model.save(os.path.join(model_dir, "final_model"))
        print(f"最終模型已儲存至 {model_dir}")
        env.close()
        eval_env.close()

    print("===== 實驗結束 =====")
    print("你可以使用以下指令來查看訓練曲線:")
    print(f"tensorboard --logdir {os.path.join('logs', experiment_name)}")
