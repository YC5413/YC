"""
訓練主程式 (整合優化版 + Death-Zone Boost)
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
from typing import Optional
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

        return max(base_eps, self.cfg['boost_epsilon'])
        
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
            agent.memorize(state, action, scaled_reward, next_state, done)

            loss = agent.update()
            if loss is not None:
                loss_epoch.append(loss)

            state = next_state

        # --- Episode 結束 ---
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

        # 計算成功率
        flag_success_count = sum(flag_obtained_history[-100:]) if flag_obtained_history else 0
        total_flags = sum(flag_obtained_history) if flag_obtained_history else 0

        flag_status = "Y" if flag_obtained else "."
        boost_status = f"B{dzb.boost_remaining:3d}" if dzb.boost_remaining > 0 else (f"CD{dzb.cooldown_remaining:3d}" if dzb.cooldown_remaining > 0 else "   ---")
        dz_str = f"{dz:.1f}" if dz is not None else "  N/A"
        bs_str = f"{bs:.1f}" if bs is not None else "  N/A"
        disabled_str = " [DISABLED]" if dzb.boost_disabled_forever else ""

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
            f"DZ:{dz_str} BS:{bs_str} {boost_status}{disabled_str}"
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

    train()
