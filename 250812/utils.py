#!/usr/bin/env python3
import numpy as np
import torch
import gym
import collections
import cv2
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# ---------------- Environment Wrappers ----------------
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self.frame_buffer = collections.deque(maxlen=2)

    def step(self, action):
        total_reward, done = 0.0, False
        info = {}
        flag_got = False
        
        for i in range(self.skip):
            obs, reward, done, step_info = self.env.step(action)
            self.frame_buffer.append(obs)
            total_reward += reward
            if 'flag_get' in step_info and step_info['flag_get']:
                flag_got = True
            if i == self.skip - 1 or done:
                info = step_info
            
            if done:
                break
        if flag_got:
            info['flag_get'] = True
        
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[-1])
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        self.frame_buffer.clear()
        obs = self.env.reset(**kwargs)
        self.frame_buffer.append(obs)
        return obs

class Grayscale(gym.ObservationWrapper):

    def __init__(self, env, training=True):
        super().__init__(env)
        self.training = training
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, 84, 84), dtype=np.float32)

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(obs, (84, 110), interpolation=cv2.INTER_LINEAR)
        cropped = resized[18:102, :]
        cropped = np.expand_dims(cropped, axis=0)

        cropped = cropped.astype(np.float32) / 255.0
        return cropped
class GrayscaleAndEdgeEnhance(gym.ObservationWrapper):
    def __init__(self, env, edge_strength=0.2):
        super().__init__(env)
        self.edge_strength = edge_strength
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, 84, 84), dtype=np.float32)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_LINEAR)
        cropped = resized[18:102, :]
        sobelx = cv2.Sobel(cropped, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(cropped, cv2.CV_32F, 0, 1, ksize=3)
        edge_map = np.sqrt(sobelx**2 + sobely**2)
        edge_map = np.clip(edge_map, 0, 255)
        edge_map = (edge_map / (edge_map.max() + 1e-8)) * 255.0
        enhanced = (1.0 - self.edge_strength) * cropped + self.edge_strength * edge_map
        enhanced = np.clip(enhanced, 0, 255)
        enhanced = np.expand_dims(enhanced, axis=0)
        enhanced = enhanced.astype(np.float32) / 255.0

        return enhanced

class FrameStackLazy(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = collections.deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(k, shp[1], shp[2]), dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return np.concatenate(list(self.frames), axis=0)

# 添加一個包裝器使render可以返回畫面數據
class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def render(self, mode='human'):
        """支持rgb_array模式，返回畫面數據"""
        if mode == 'rgb_array':
            # 直接使用環境的rgb_array模式渲染
            try:
                # 嘗試直接使用原始環境的rgb_array渲染
                return self.env.render(mode='rgb_array')
            except Exception as e:
                print(f"使用rgb_array模式渲染時出錯: {e}")
                # 如果無法直接渲染，返回空數據
                return np.zeros((240, 256, 3), dtype=np.uint8)
        else:
            # 正常渲染
            return self.env.render(mode)

def create_mario_env(training=True, render_mode=None):
    # 舊版本的gym不支持在make時指定render_mode
    env = gym_super_mario_bros.make('SuperMarioBros-v0')  
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MaxAndSkipEnv(env, skip=4)
    env = Grayscale(env, training=training)
    env = FrameStackLazy(env, k=4)
    env = RenderWrapper(env)  # 添加渲染包裝器
    return env

# ---------------- Utility Functions ----------------
def plot_frames(state, title=""):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axs[i].imshow(state[i], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'Frame {i+1}')
    plt.suptitle(title)
    plt.show()
