"""
訓練主程式 (整合優化版 + Death-Zone Boost + Replay Bonus)
支援五種 DQN 演算法：DQN, Double DQN, Dueling DQN, QR-DQN, QR-D3QN
包含進階功能：
- Dynamic Gamma (Curriculum Learning)
- Dynamic Learning Rate (Scheduler)
- Death-Zone Epsilon Boost
  * 統計每次死亡的 x 位置（death_x_buf，滑動窗口）
  * death_zone = p85(death_x_buf)
  * 當 p85 - p50 < boost_spread_threshold 時觸發 boost
  * boost 生效區間：[death_zone - boost_x_offset, ∞)（不低於 boost_min_x）
  * boost 結束後進 cooldown，cooldown 後可再 arm
  * 若已通關且 death_zone >= clear_x - clear_x_margin，永久停用 boost
- Replay Bonus (Death-Zone 成功穿越片段加權回放)
  * episode 結束後統一判定是否成功穿越 death zone
  * 成功 episode 中 death zone 附近的 transition 標記為 bonus
  * PER priority 乘以衰退的 bonus factor
  * bonus 隨 episode age 衰退，超過 max age 自動回歸一般權重
"""
import torch
import torch.optim as optim
import numpy as np
import random
import os
import cv2
import datetime
import argparse
from collections import deque
from typing import Optional, List, Dict, Any
from torch.utils.tensorboard import SummaryWriter

from utils import create_mario_env, get_shaped_reward
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from qrdqn_agent import QRDQNAgent
from qrd3qn_agent import QRD3QNAgent

# ==================== Configuration ====================

CONFIG = {
    # 演算法選擇: 'dqn', 'double_dqn', 'dueling_dqn', 'qrdqn', 'qrd3qn'
    'algorithm': 'qrd3qn',

    # 隨機種子設置
    'seed': 42,

    # 環境設置
    'env_name': 'SuperMarioBros-1-3-v0',
    'world': 1,
    'stage': 3,
    'skip_frames': 4,
    'frame_stack': 4,

    # 訓練基本設定
    'num_episodes': 40000,
    'batch_size': 64,
    'learning_rate': 0.000125,
    'max_memory_size': 200000,
    'target_update_freq': 500,

    # 動態 Gamma 設定 (Curriculum Learning)
    'gamma_start': 0.95,
    'gamma_end': 0.95,
    'gamma_anneal_episodes': 40000,

    # Epsilon 基本設定
    'epsilon_start': 1.0,
    'epsilon_min': 0.015,
    'epsilon_decay': 0.9975,

    # Learning Rate Scheduler 設定
    'lr_max_decays': 2,
    'lr_factor': 1,
    'lr_decay_interval': 6000,

    # QR-DQN / QR-D3QN 專用參數
    'num_quantiles': 51,
    'kappa': 1.0,

    # Reward Shaping 獎勵塑形設定
    'flag_bonus': 200,
    'forward_reward': 2,
    'penalty': -0.5,
    'reward_scaling': 200,

    # ==================== Death-Zone Boost ====================
    # 死亡位置統計
    'death_zone_window': 150,       # 滑動窗口：記錄最近 N 次死亡的 x 位置
    'death_zone_min_samples': 30,   # 至少累積這麼多樣本才計算 death_zone

    # Boost 觸發條件
    'boost_spread_threshold': 100.0, # p85 - p50 < threshold 才觸發

    # Boost 行為
    'boost_epsilon': 0.35,          # boost 生效時的 epsilon 下限
    'boost_duration': 200,          # boost 持續的 episode 數
    'boost_cooldown': 100,          # boost 結束後的冷卻 episode 數

    # Boost 生效區間（x 座標範圍）
    'boost_x_offset': 200.0,        # boost 起點 = death_zone - boost_x_offset
    'boost_min_x': 300.0,           # boost 起點的絕對下限

    # 永久停用條件
    'clear_x_margin': 5.0,          # death_zone >= clear_x - margin 時永久停用（保留但不再使用）
    'boost_disable_on_clear_count': 3,  # 累積通關 N 次後永久停用 boost
    # =========================================================

    # ==================== Replay Bonus ====================
    'replay_bonus_enabled': True,       # 總開關，方便一鍵關閉做對照實驗
    'dz_bonus_window': 200,             # bonus 範圍：death zone ± 200 的 transition
    'dz_success_margin': 120,           # 成功穿越條件：episode_max_x >= death_zone + 120
    'dz_bonus_factor': 3.0,             # priority 乘法加權倍數
    'dz_bonus_max_age': 3000,           # 超過 3000 episodes 不再保留額外 bonus
    'dz_bonus_decay_gamma': 0.999,      # 每增加 1 episode age，bonus 乘上 0.999
    'dz_bonus_min_factor': 1.0,         # 衰退後最低回到一般樣本
    # ======================================================

    # 模型保存與視頻錄製設定
    'checkpoint_dir': 'checkpoints',
    'save_interval': 1000,
    'video_interval': 500,
    'video_dir': 'videos',

    # 預訓練模型載入設定
    'load_model': False,
    'model_path': None,
    'postfix': '',
    'xpos_save_threshold': None,
}


# ==================== Death-Zone Boost State ====================

class DeathZoneBoost:
    """
    追蹤死亡 x 位置並管理 epsilon boost 狀態。

    每次 Mario 扣命（life < last_life）時呼叫 record_death(x)。
    每個 episode 結束時呼叫 end_episode() 更新 boost/cooldown 計數。
    在 episode 內每步呼叫 effective_epsilon(base_eps, x) 取得實際 epsilon。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.death_x_buf: deque = deque(maxlen=cfg['death_zone_window'])
        self.boost_remaining: int = 0
        self.cooldown_remaining: int = 0
        self.boost_count: int = 0
        self.boost_disabled_forever: bool = False
        self.cleared_once: bool = False
        self.clear_x: Optional[float] = None
        self.flag_count: int = 0

    # ------------------------------------------------------------------
    # 外部呼叫介面
    # ------------------------------------------------------------------

    def record_death(self, x: float):
        """每次扣命時記錄死亡 x 座標。"""
        self.death_x_buf.append(float(x))

    def record_flag(self, x: float):
        """通關時更新 clear_x，累積 N 次通關後永久停用 boost。"""
        self.cleared_once = True
        self.clear_x = float(x)  # 持續更新，取最新通關 x
        self.flag_count += 1
        if self.flag_count >= self.cfg['boost_disable_on_clear_count']:
            self.boost_disabled_forever = True
            self.boost_remaining = 0
            self.cooldown_remaining = 0

    def end_episode(self):
        """Episode 結束時推進 boost/cooldown 計數，並檢查是否應觸發 boost。"""
        if self.boost_remaining > 0:
            self.boost_remaining -= 1
            if self.boost_remaining == 0:
                self.cooldown_remaining = self.cfg['boost_cooldown']
        elif self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        # 嘗試觸發 boost（只在 idle 狀態下）
        if self._should_trigger():
            self._trigger()

    def effective_epsilon(self, base_eps: float, x: float) -> float:
        """
        回傳該 step 實際使用的 epsilon。
        只有 boost active 且 x 在 boost zone 內才提高 epsilon。
        """
        if self.boost_disabled_forever:
            return base_eps
        if self.boost_remaining <= 0:
            return base_eps

        boost_start = self._compute_boost_start()
        if boost_start is None:
            return base_eps
        if x >= boost_start:
            return max(base_eps, self.cfg['boost_epsilon'])
        return base_eps

    # ------------------------------------------------------------------
    # 查詢介面（用於 log）
    # ------------------------------------------------------------------

    @property
    def death_zone(self) -> Optional[float]:
        """p85 of death_x_buf，樣本不足時回傳 None。"""
        return self._compute_death_zone()

    @property
    def boost_start(self) -> Optional[float]:
        return self._compute_boost_start()

    @property
    def is_boost_active(self) -> bool:
        return self.boost_remaining > 0

    # ------------------------------------------------------------------
    # 內部方法
    # ------------------------------------------------------------------

    def _compute_death_zone(self) -> Optional[float]:
        if len(self.death_x_buf) < self.cfg['death_zone_min_samples'] and not self.cleared_once:
            return None
        if not self.death_x_buf:
            return None
        return float(np.percentile(list(self.death_x_buf), 85))

    def _compute_boost_start(self) -> Optional[float]:
        dz = self._compute_death_zone()
        if dz is None:
            return None
        if self.boost_disabled_forever:
            return None
        return max(self.cfg['boost_min_x'], dz - self.cfg['boost_x_offset'])

    def _should_trigger(self) -> bool:
        if self.boost_disabled_forever:
            return False
        if self.boost_remaining > 0 or self.cooldown_remaining > 0:
            return False

        values = list(self.death_x_buf)
        if len(values) < self.cfg['death_zone_min_samples']:
            return False

        p85 = float(np.percentile(values, 85))
        p50 = float(np.percentile(values, 50))
        spread = p85 - p50
        return spread < self.cfg['boost_spread_threshold']

    def _trigger(self):
        self.boost_remaining = self.cfg['boost_duration']
        self.boost_count += 1

# ==================== Replay Bonus Manager ====================

class ReplayBonusManager:
    """
    管理 death-zone 成功穿越的 replay bonus 機制。

    在 episode 結束後統一判定是否成功穿越 death zone，
    並對 death zone 附近的 transition 標記 bonus metadata。
    在 agent 從 replay buffer 取樣時，透過修改 priority
    來提高這些 transition 的被抽到的機會。

    本 manager 不修改 replay buffer 的儲存結構，
    而是維護一個獨立的 metadata 表來追蹤 bonus 資訊。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = cfg.get('replay_bonus_enabled', True)

        # bonus metadata：key = buffer index, value = metadata dict
        # 由於 replay buffer 是環形的，舊 entry 會被覆蓋，
        # 我們用 episode-based 的 list 來追蹤 bonus samples
        self.bonus_records: List[Dict[str, Any]] = []

        # 統計用
        self.total_success_crosses = 0
        self.total_bonus_transitions = 0

    def get_effective_bonus_factor(self, is_dz_bonus: bool,
                                   bonus_factor: float,
                                   insert_episode: int,
                                   current_episode: int) -> float:
        """
        計算某筆 transition 的 effective bonus factor。

        公式：
        - 若 is_dz_bonus = False → 1.0
        - 若 age > dz_bonus_max_age → 1.0
        - 否則 → max(dz_bonus_min_factor, bonus_factor * decay_gamma^age)
        """
        if not self.enabled or not is_dz_bonus:
            return 1.0

        age = current_episode - insert_episode
        if age > self.cfg['dz_bonus_max_age']:
            return 1.0

        decayed = bonus_factor * (self.cfg['dz_bonus_decay_gamma'] ** age)
        return max(self.cfg['dz_bonus_min_factor'], decayed)

    def evaluate_episode(self, episode_buffer: List[Dict],
                         episode_id: int,
                         episode_max_x: float,
                         current_death_zone: Optional[float]) -> List[Dict]:
        """
        在 episode 結束後判定是否成功穿越 death zone，
        並為 episode_buffer 中的每筆 transition 標記 bonus metadata。

        回傳帶有 bonus metadata 的 transition list。
        """
        # 若 replay bonus 關閉或 death zone 尚未計算出，全部不標記
        if not self.enabled or current_death_zone is None:
            for tr in episode_buffer:
                tr['is_dz_bonus'] = False
                tr['dz_center_at_insert'] = None
                tr['insert_episode'] = episode_id
                tr['bonus_factor'] = 1.0
            return episode_buffer

        # 判定是否成功穿越
        success_cross = (episode_max_x >= current_death_zone + self.cfg['dz_success_margin'])

        if success_cross:
            self.total_success_crosses += 1

        dz_lo = current_death_zone - self.cfg['dz_bonus_window']
        dz_hi = current_death_zone + self.cfg['dz_bonus_window']

        bonus_count = 0
        for tr in episode_buffer:
            is_bonus = (
                success_cross and
                dz_lo <= tr['x_pos'] <= dz_hi
            )
            tr['is_dz_bonus'] = is_bonus
            tr['dz_center_at_insert'] = current_death_zone if is_bonus else None
            tr['insert_episode'] = episode_id
            tr['bonus_factor'] = self.cfg['dz_bonus_factor'] if is_bonus else 1.0

            if is_bonus:
                bonus_count += 1

        self.total_bonus_transitions += bonus_count
        return episode_buffer


# ==================== Bonus-Aware Replay Buffer Wrapper ====================

class BonusAwareReplayBuffer:
    """
    包裝 agent 的 replay buffer，為每筆 transition 附加 bonus metadata。

    本 wrapper 維護一個與 agent.memory 平行的 metadata 陣列，
    並在 agent.update() 前攔截 sampling，將 bonus factor 乘入 priority。

    設計原則：
    - 不修改 agent 內部的 replay buffer 結構
    - 透過 monkey-patch agent 的 memorize / update 方法注入 bonus 邏輯
    - metadata 使用環形 buffer 與 agent.memory 同步
    """

    def __init__(self, agent, cfg: dict, bonus_manager: ReplayBonusManager):
        self.agent = agent
        self.cfg = cfg
        self.bonus_manager = bonus_manager
        self.max_size = cfg['max_memory_size']

        # 平行 metadata buffer（與 agent.memory 同步）
        self.metadata: List[Optional[Dict]] = [None] * self.max_size
        self.write_idx = 0
        self.current_size = 0

        # 保存原始方法
        self._original_memorize = agent.memorize
        self._original_update = agent.update

        # 當前 episode（用於計算 age）
        self.current_episode = 0

        # 統計：最近一次 batch 中的 bonus 比例
        self.last_batch_bonus_ratio = 0.0
        self.last_batch_avg_bonus_factor = 1.0
        self.buffer_bonus_count = 0

    def memorize_with_metadata(self, state, action, reward, next_state, done,
                                is_dz_bonus=False, dz_center_at_insert=None,
                                insert_episode=0, bonus_factor=1.0):
        """
        呼叫原始 memorize 並同步寫入 metadata。
        """
        # 呼叫原始 memorize
        self._original_memorize(state, action, reward, next_state, done)

        # 寫入 metadata
        meta = {
            'is_dz_bonus': is_dz_bonus,
            'dz_center_at_insert': dz_center_at_insert,
            'insert_episode': insert_episode,
            'bonus_factor': bonus_factor,
        }

        # 更新 bonus count（維護 buffer 中 bonus 總數）
        old_meta = self.metadata[self.write_idx]
        if old_meta is not None and old_meta.get('is_dz_bonus', False):
            self.buffer_bonus_count -= 1
        if is_dz_bonus:
            self.buffer_bonus_count += 1

        self.metadata[self.write_idx] = meta
        self.write_idx = (self.write_idx + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def update_with_bonus(self) -> Optional[float]:
        """
        執行 agent.update()，但在 sampling 前修改 priority。

        由於不同 agent 實作的 replay buffer 結構不同，
        這裡採用最通用的策略：

        方案 A（有 PER）：修改 agent.memory 中的 priorities
        方案 B（無 PER）：直接呼叫原始 update，bonus 不生效

        為了保持最大相容性，這裡偵測 agent.memory 是否有 priorities 屬性。
        """
        if not self.cfg.get('replay_bonus_enabled', True):
            return self._original_update()

        # 偵測 agent 是否使用 PER
        memory = self.agent.memory if hasattr(self.agent, 'memory') else None

        if memory is not None and hasattr(memory, 'priorities') and hasattr(memory, 'position'):
            # PER replay buffer: 在 sampling 前修改 priorities
            self._apply_bonus_to_priorities(memory)

        loss = self._original_update()

        # 若 PER 在 update 後會重新計算 priority（td error based），
        # 我們在下次 update 時再次 apply bonus
        return loss

    def _apply_bonus_to_priorities(self, memory):
        """
        將 bonus factor 乘入 PER 的 priorities。

        注意：PER 通常會在 update 後用 td error 更新 priority，
        所以我們每次 update 前都需要重新 apply。
        我們不直接修改 base priority，而是在取樣時透過
        暫時調高 priority 來實現加權效果。
        """
        # 取得目前 buffer 的有效大小
        buf_size = min(len(memory) if hasattr(memory, '__len__') else self.current_size,
                       self.max_size)

        if buf_size == 0:
            return

        # 取得 priorities 陣列（可能是 numpy array 或 SumTree）
        priorities = None
        if hasattr(memory, 'priorities') and isinstance(memory.priorities, np.ndarray):
            priorities = memory.priorities
        elif hasattr(memory, 'tree') and hasattr(memory.tree, 'tree'):
            # SumTree 結構
            priorities = None  # SumTree 不直接暴露 priorities，見下方處理

        if priorities is not None:
            # 直接修改 priorities 陣列
            bonus_count = 0
            total_factor = 0.0
            for i in range(buf_size):
                meta = self.metadata[i]
                if meta is not None and meta.get('is_dz_bonus', False):
                    factor = self.bonus_manager.get_effective_bonus_factor(
                        is_dz_bonus=True,
                        bonus_factor=meta['bonus_factor'],
                        insert_episode=meta['insert_episode'],
                        current_episode=self.current_episode,
                    )
                    if factor > 1.0:
                        priorities[i] *= factor
                        bonus_count += 1
                        total_factor += factor

            if bonus_count > 0:
                self.last_batch_avg_bonus_factor = total_factor / bonus_count
            else:
                self.last_batch_avg_bonus_factor = 1.0

    def set_current_episode(self, episode: int):
        self.current_episode = episode


# ==================== Utility Functions ====================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== Agent Creation ====================

def create_agent(state_shape, n_actions, algorithm):
    common_params = {
        'state_shape': state_shape,
        'n_actions': n_actions,
        'lr': CONFIG['learning_rate'],
        'gamma': CONFIG['gamma_start'],
        'batch_size': CONFIG['batch_size'],
        'max_memory_size': CONFIG['max_memory_size'],
        'target_update_freq': CONFIG['target_update_freq'],
    }

    if algorithm == 'dqn':
        agent = DQNAgent(**common_params)
    elif algorithm == 'double_dqn':
        agent = DoubleDQNAgent(**common_params)
    elif algorithm == 'dueling_dqn':
        agent = DuelingDQNAgent(**common_params)
    elif algorithm == 'qrdqn':
        agent = QRDQNAgent(**common_params,
                           num_quantiles=CONFIG['num_quantiles'],
                           kappa=CONFIG['kappa'])
    elif algorithm == 'qrd3qn':
        agent = QRD3QNAgent(**common_params,
                            num_quantiles=CONFIG['num_quantiles'],
                            kappa=CONFIG['kappa'])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if CONFIG['load_model'] and CONFIG['model_path'] and os.path.exists(CONFIG['model_path']):
        if CONFIG.get('learning_rate') is not None:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = CONFIG['learning_rate']
            print(f"[Override] Resume: optimizer learning rate set to {CONFIG['learning_rate']}")

    return agent


# ==================== Training Loop ====================

def train():
    set_seed(CONFIG['seed'])

    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['video_dir'], exist_ok=True)

    env = create_mario_env(
        env_name=CONFIG['env_name'],
        skip=CONFIG['skip_frames'],
        frame_stack=CONFIG['frame_stack']
    )

    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = create_agent(state_shape, n_actions, CONFIG['algorithm'])

    # Death-Zone Boost 狀態機
    dzb = DeathZoneBoost(CONFIG)

    # Replay Bonus Manager
    rb_manager = ReplayBonusManager(CONFIG)

    # Bonus-Aware Replay Buffer Wrapper
    bonus_buffer = BonusAwareReplayBuffer(agent, CONFIG, rb_manager)

    lr_decay_count = 0

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    env_tag = f"w{CONFIG.get('world', 'x')}{CONFIG.get('stage', 'x')}"
    writer = SummaryWriter(f"runs/{CONFIG['algorithm']}_{env_tag}_{current_time}")
    log_file = f"training_log_{CONFIG['algorithm']}_{env_tag}_{current_time}.txt"

    def log_print(message):
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    import sys
    log_print(f"Command line: {' '.join(sys.argv)}")
    log_print("=" * 80)
    log_print(f"Training started at {current_time}")
    log_print(f"Algorithm: {CONFIG['algorithm'].upper()}")
    log_print(f"Environment: {CONFIG['env_name']}")
    log_print(f"Random Seed: {CONFIG['seed']}")
    if CONFIG.get('load_model') and CONFIG.get('model_path'):
        if os.path.exists(CONFIG['model_path']):
            log_print(f"Resuming from checkpoint: {CONFIG['model_path']}")
        else:
            log_print(f"Requested resume from missing checkpoint: {CONFIG['model_path']}")
    else:
        log_print("Starting training from scratch (no resume)")
    log_print(f"Total episodes: {CONFIG['num_episodes']}")
    log_print(f"Batch size: {CONFIG['batch_size']}, Learning rate: {CONFIG['learning_rate']}")
    log_print(f"Memory size: {CONFIG['max_memory_size']}, Target update freq: {CONFIG['target_update_freq']}")
    if CONFIG['algorithm'] in ['qrdqn', 'qrd3qn']:
        log_print(f"Quantiles: {CONFIG['num_quantiles']}, Kappa: {CONFIG['kappa']}")
    log_print(f"Config: Gamma {CONFIG['gamma_start']}->{CONFIG['gamma_end']}, Epsilon {CONFIG['epsilon_start']}->{CONFIG['epsilon_min']}")
    log_print(
        f"Death-Zone Boost: window={CONFIG['death_zone_window']} min_samples={CONFIG['death_zone_min_samples']} "
        f"spread_thresh={CONFIG['boost_spread_threshold']} boost_eps={CONFIG['boost_epsilon']} "
        f"duration={CONFIG['boost_duration']} cooldown={CONFIG['boost_cooldown']} "
        f"x_offset={CONFIG['boost_x_offset']} min_x={CONFIG['boost_min_x']}"
    )
    log_print(
        f"Replay Bonus: enabled={CONFIG['replay_bonus_enabled']} "
        f"bonus_window={CONFIG['dz_bonus_window']} success_margin={CONFIG['dz_success_margin']} "
        f"bonus_factor={CONFIG['dz_bonus_factor']} max_age={CONFIG['dz_bonus_max_age']} "
        f"decay_gamma={CONFIG['dz_bonus_decay_gamma']} min_factor={CONFIG['dz_bonus_min_factor']}"
    )
    log_print("=" * 80)

    # 訓練統計
    total_rewards = []
    total_orig_rewards = []
    x_max_positions = []
    flag_obtained_history = []

    if CONFIG.get('load_model') and CONFIG.get('model_path'):
        epsilon = CONFIG['epsilon_min']
    else:
        epsilon = CONFIG['epsilon_start']

    global_max_x = 0
    xpos_threshold_triggered = False

    # 訓練循環
    for episode in range(CONFIG['num_episodes']):

        # 設定當前 episode（用於 bonus age 計算）
        bonus_buffer.set_current_episode(episode)

        # 動態 Gamma 更新 (Annealing)
        gamma_progress = min(1.0, episode / CONFIG['gamma_anneal_episodes'])
        current_gamma = CONFIG['gamma_start'] + (CONFIG['gamma_end'] - CONFIG['gamma_start']) * gamma_progress
        agent.gamma = current_gamma

        state = torch.from_numpy(env.reset()).float()
        total_reward = 0
        total_orig_reward = 0
        done = False
        x_max_pos = 0
        flag_obtained = False
        loss_epoch = []

        last_x = 0

        # 本 episode boost 監控（用於 log）
        boost_was_active_this_ep = False

        # ===== Replay Bonus: episode 暫存區 =====
        episode_buffer: List[Dict[str, Any]] = []

        # 錄製視頻
        record_video = (episode + 1) % CONFIG['video_interval'] == 0
        if record_video:
            video_path = os.path.join(CONFIG['video_dir'], f"{env_tag}_ep_{episode+1}_{CONFIG['algorithm']}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (256, 240))
            frames = []

        # Episode 循環
        while not done:
            # 錄製幀
            if record_video:
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frames.append(frame_bgr)

            # 取得當前 x 座標（用於 boost zone 判斷）
            # 注意：info 在 step 之後才有，這裡用上一步的 last_x 做 zone 判斷
            eff_epsilon = dzb.effective_epsilon(epsilon, last_x)
            if eff_epsilon > epsilon:
                boost_was_active_this_ep = True

            # 選擇動作
            action = agent.select_action(state, eff_epsilon)

            # 執行動作
            next_obs, reward, done, info = env.step(int(action))

            # 更新 x
            x_now = float(info.get('x_pos', last_x))

            # 通關偵測
            if info.get('flag_get', False):
                flag_obtained = True
                dzb.record_flag(x_now)

            # 死亡偵測：此環境 life 恆為 2，改用 episode 結束且未通關判斷
            # done=True and not flag_obtained 代表 Mario 死亡或超時
            if done and not flag_obtained:
                dzb.record_death(x_now)

            last_x = x_now

            # Reward Shaping
            shaped_reward, x_max_pos = get_shaped_reward(
                reward, info, x_max_pos,
                flag_bonus=CONFIG['flag_bonus'],
                forward_reward=CONFIG['forward_reward'],
                penalty=CONFIG['penalty']
            )

            total_orig_reward += reward
            scaled_reward = shaped_reward / CONFIG['reward_scaling']

            total_reward += shaped_reward

            next_state = torch.from_numpy(next_obs).float()

            # ===== Replay Bonus: 暫存到 episode_buffer，不直接 memorize =====
            episode_buffer.append({
                'state': state,
                'action': action,
                'reward': scaled_reward,
                'next_state': next_state,
                'done': done,
                'x_pos': x_now,
                'episode_id': episode,
            })

            # 每步仍然呼叫 update（從 replay buffer 中已有的資料學習）
            loss = bonus_buffer.update_with_bonus()
            if loss is not None:
                loss_epoch.append(loss)

            state = next_state

        # --- Episode 結束 ---

        # ===== Replay Bonus: 判定成功穿越並標記 bonus，然後批次寫入 replay buffer =====
        current_death_zone = dzb.death_zone
        episode_buffer = rb_manager.evaluate_episode(
            episode_buffer=episode_buffer,
            episode_id=episode,
            episode_max_x=x_max_pos,
            current_death_zone=current_death_zone,
        )

        # 統計本回合 bonus transition 數量
        ep_bonus_count = sum(1 for tr in episode_buffer if tr.get('is_dz_bonus', False))
        ep_success_cross = (
            current_death_zone is not None and
            x_max_pos >= current_death_zone + CONFIG['dz_success_margin']
        )

        # 批次寫入 replay buffer
        for tr in episode_buffer:
            bonus_buffer.memorize_with_metadata(
                state=tr['state'],
                action=tr['action'],
                reward=tr['reward'],
                next_state=tr['next_state'],
                done=tr['done'],
                is_dz_bonus=tr.get('is_dz_bonus', False),
                dz_center_at_insert=tr.get('dz_center_at_insert'),
                insert_episode=tr.get('insert_episode', episode),
                bonus_factor=tr.get('bonus_factor', 1.0),
            )

        # 通知 DeathZoneBoost episode 結束（更新 boost/cooldown，嘗試觸發）
        dzb.end_episode()

        # 保存視頻
        if record_video and frames:
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()
            log_print(f"Video saved: {video_path}")

        # 數據統計
        avg_loss = np.mean(loss_epoch) if loss_epoch else 0
        total_rewards.append(total_reward)
        total_orig_rewards.append(total_orig_reward)
        x_max_positions.append(x_max_pos)
        flag_obtained_history.append(flag_obtained)

        if x_max_pos > global_max_x:
            global_max_x = x_max_pos

        avg_x_100 = np.mean(x_max_positions[-100:]) if len(x_max_positions) >= 100 else np.mean(x_max_positions)
        avg_reward_100 = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)

        # xpos 閾值一次性存檔
        xpos_threshold = CONFIG.get('xpos_save_threshold')
        if xpos_threshold is not None and not xpos_threshold_triggered:
            try:
                if avg_x_100 >= float(xpos_threshold):
                    postfix = f"_{CONFIG['postfix']}" if CONFIG.get('postfix') else ''
                    cp_path = os.path.join(
                        CONFIG['checkpoint_dir'],
                        f"{CONFIG['algorithm']}_{env_tag}_ep{episode+1}_xpos{int(xpos_threshold)}{postfix}.pt"
                    )
                    agent.save(cp_path)
                    log_print(f"Checkpoint saved (xpos threshold reached): {cp_path}")
                    xpos_threshold_triggered = True
            except Exception as e:
                log_print(f"Error during xpos-threshold checkpoint save: {e}")

        # LR decay
        if lr_decay_count < CONFIG['lr_max_decays'] and (episode + 1) % CONFIG['lr_decay_interval'] == 0:
            old_lr = agent.optimizer.param_groups[0]['lr']
            for pg in agent.optimizer.param_groups:
                pg['lr'] *= CONFIG['lr_factor']
            lr_decay_count += 1
            log_print(f"LR decayed ({lr_decay_count}/{CONFIG['lr_max_decays']}): {old_lr:.2e} -> {agent.optimizer.param_groups[0]['lr']:.2e}")

        # Epsilon 單調衰減
        epsilon = max(CONFIG['epsilon_min'], epsilon * CONFIG['epsilon_decay'])

        current_lr = agent.optimizer.param_groups[0]['lr']

        # TensorBoard
        writer.add_scalar('Reward/Total', total_reward, episode)
        writer.add_scalar('Reward/Original', total_orig_reward, episode)
        writer.add_scalar('Loss/Average', avg_loss, episode)
        writer.add_scalar('Performance/Max_X', x_max_pos, episode)
        writer.add_scalar('Performance/Avg_X_100', avg_x_100, episode)
        writer.add_scalar('Params/Epsilon', epsilon, episode)
        writer.add_scalar('Params/LearningRate', current_lr, episode)
        writer.add_scalar('Params/Gamma', current_gamma, episode)
        dz = dzb.death_zone
        bs = dzb.boost_start
        writer.add_scalar('DeathZone/DeathZone_X', dz if dz is not None else 0, episode)
        writer.add_scalar('DeathZone/BoostStart_X', bs if bs is not None else 0, episode)
        writer.add_scalar('DeathZone/BoostRemaining', dzb.boost_remaining, episode)
        writer.add_scalar('DeathZone/DeathSamples', len(dzb.death_x_buf), episode)

        # Replay Bonus TensorBoard
        writer.add_scalar('ReplayBonus/EpBonusTransitions', ep_bonus_count, episode)
        writer.add_scalar('ReplayBonus/EpSuccessCross', int(ep_success_cross), episode)
        writer.add_scalar('ReplayBonus/TotalSuccessCrosses', rb_manager.total_success_crosses, episode)
        writer.add_scalar('ReplayBonus/TotalBonusTransitions', rb_manager.total_bonus_transitions, episode)
        writer.add_scalar('ReplayBonus/BufferBonusCount', bonus_buffer.buffer_bonus_count, episode)
        writer.add_scalar('ReplayBonus/AvgBonusFactor', bonus_buffer.last_batch_avg_bonus_factor, episode)

        # 計算成功率
        flag_success_count = sum(flag_obtained_history[-100:]) if flag_obtained_history else 0
        total_flags = sum(flag_obtained_history) if flag_obtained_history else 0

        flag_status = "Y" if flag_obtained else "."
        boost_status = f"B{dzb.boost_remaining:3d}" if dzb.boost_remaining > 0 else (f"CD{dzb.cooldown_remaining:3d}" if dzb.cooldown_remaining > 0 else "   ---")
        dz_str = f"{dz:.1f}" if dz is not None else "  N/A"
        bs_str = f"{bs:.1f}" if bs is not None else "  N/A"
        disabled_str = " [DISABLED]" if dzb.boost_disabled_forever else ""

        # Replay Bonus 狀態字串
        rb_str = f"RB:{ep_bonus_count:3d}" if ep_success_cross else "RB:  -"

        log_print(
            f"Ep {episode+1:4d} |"
            f"OrigR:{total_orig_reward:6.1f} ShpR:{total_reward:6.1f} |"
            f"X:{x_max_pos:6.1f}({global_max_x:6.1f}) AvgX100:{avg_x_100:6.1f} |"
            f"AvgR:{avg_reward_100:6.1f} |"
            f"Ls:{avg_loss:.4f} |"
            f"Eps:{epsilon:.4f} |"
            f"LR:{current_lr:.2e} |"
            f"Gm:{current_gamma:.3f} |"
            f"Fl:{flag_status}({flag_success_count}/100){total_flags} |"
            f"DZ:{dz_str} BS:{bs_str} {boost_status}{disabled_str} |"
            f"{rb_str}"
        )

        # 定期 checkpoint
        if (episode + 1) % CONFIG['save_interval'] == 0:
            postfix = f"_{CONFIG['postfix']}" if CONFIG.get('postfix') else ''
            checkpoint_path = os.path.join(
                CONFIG['checkpoint_dir'],
                f"{CONFIG['algorithm']}_{env_tag}_ep{episode+1}{postfix}.pt"
            )
            agent.save(checkpoint_path)
            log_print(f"Checkpoint saved: {checkpoint_path}")

    # 最終保存
    postfix = f"_{CONFIG['postfix']}" if CONFIG.get('postfix') else ''
    final_path = os.path.join(
        CONFIG['checkpoint_dir'],
        f"{CONFIG['algorithm']}_{env_tag}_final{postfix}.pt"
    )
    agent.save(final_path)
    log_print(f"Final model saved: {final_path}")

    # Replay Bonus 最終統計
    log_print(f"Replay Bonus Summary: total_success_crosses={rb_manager.total_success_crosses}, "
              f"total_bonus_transitions={rb_manager.total_bonus_transitions}")

    writer.close()
    env.close()
    log_print("Training completed!")


# ==================== Main ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Mario Bros DQN Agent')
    parser.add_argument('--algorithm', type=str, default='qrd3qn',
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'qrdqn', 'qrd3qn'])
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--world', type=int, default=None)
    parser.add_argument('--stage', type=int, default=None)
    parser.add_argument('--postfix', type=str, default=None)
    parser.add_argument('--xpos_save_threshold', type=float, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--load_model', type=bool, default=False)
    # Death-Zone Boost 參數（可從命令列覆蓋）
    parser.add_argument('--death_zone_window', type=int, default=None)
    parser.add_argument('--death_zone_min_samples', type=int, default=None)
    parser.add_argument('--boost_spread_threshold', type=float, default=None)
    parser.add_argument('--boost_epsilon', type=float, default=None)
    parser.add_argument('--boost_duration', type=int, default=None)
    parser.add_argument('--boost_cooldown', type=int, default=None)
    parser.add_argument('--boost_x_offset', type=float, default=None)
    parser.add_argument('--boost_min_x', type=float, default=None)
    parser.add_argument('--clear_x_margin', type=float, default=None)
    # Replay Bonus 參數（可從命令列覆蓋）
    parser.add_argument('--replay_bonus_enabled', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--dz_bonus_window', type=int, default=None)
    parser.add_argument('--dz_success_margin', type=int, default=None)
    parser.add_argument('--dz_bonus_factor', type=float, default=None)
    parser.add_argument('--dz_bonus_max_age', type=int, default=None)
    parser.add_argument('--dz_bonus_decay_gamma', type=float, default=None)
    parser.add_argument('--dz_bonus_min_factor', type=float, default=None)

    args = parser.parse_args()

    if args.world is not None and args.stage is not None:
        CONFIG['env_name'] = f'SuperMarioBros-{args.world}-{args.stage}-v0'
        CONFIG['world'] = args.world
        CONFIG['stage'] = args.stage

    if args.algorithm:
        CONFIG['algorithm'] = args.algorithm
    if args.episodes:
        CONFIG['num_episodes'] = args.episodes
    if args.seed is not None:
        CONFIG['seed'] = args.seed
    if args.gamma is not None:
        CONFIG['gamma_start'] = args.gamma
        CONFIG['gamma_end'] = args.gamma
    if args.postfix is not None:
        CONFIG['postfix'] = args.postfix
    if args.xpos_save_threshold is not None:
        CONFIG['xpos_save_threshold'] = args.xpos_save_threshold
    if args.lr is not None:
        CONFIG['learning_rate'] = args.lr
    if args.model_path:
        CONFIG['model_path'] = args.model_path
        CONFIG['load_model'] = True

    # Death-Zone Boost 覆蓋
    for key in ['death_zone_window', 'death_zone_min_samples', 'boost_spread_threshold',
                'boost_epsilon', 'boost_duration', 'boost_cooldown',
                'boost_x_offset', 'boost_min_x', 'clear_x_margin']:
        val = getattr(args, key)
        if val is not None:
            CONFIG[key] = val

    # Replay Bonus 覆蓋
    for key in ['replay_bonus_enabled', 'dz_bonus_window', 'dz_success_margin',
                'dz_bonus_factor', 'dz_bonus_max_age', 'dz_bonus_decay_gamma',
                'dz_bonus_min_factor']:
        val = getattr(args, key)
        if val is not None:
            CONFIG[key] = val

    train()
