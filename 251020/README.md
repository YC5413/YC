OpenMario — 使用說明 (繁體中文)

概要

本專案包含一個使用 Stable-Baselines3 訓練 Super Mario Bros 的執行腳本。
主要腳本為 `mario.py`，會建立向量化的 Mario 環境、套用簡單的前處理，並訓練與評估強化學習代理。

需求

- Python 3.8 (64-bit)
- 若使用 GPU：需有 NVIDIA CUDA 支援的顯示卡，並安裝相容的 PyTorch wheel

安裝（PowerShell 範例）

1) 更新 pip 與相關工具：

    python -m pip install --upgrade pip setuptools wheel

2) 若使用 GPU，請先安裝對應 CUDA 版本的官方 PyTorch wheel（以下為 CUDA 11.8 範例）：

    python -m pip install torch==2.4.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3) 安裝其餘相依套件：

    python -m pip install -r requirements.txt

（若使用 conda：建議建立 Python 3.8 的 conda 環境，再透過 PyTorch 與 NVIDIA channel 安裝 pytorch，接著用 pip 安裝其他套件。）

設定（環境變數）

腳本會從環境變數讀取設定，若未設定則使用預設值：

- MARIO_MODEL：模型名稱（PPO、DQN、A2C、QRDQN）。預設：DQN
- MARIO_WORLD：世界編號。預設：1
- MARIO_STAGE：關卡編號。預設：1
- MARIO_VERSION：環境版本後綴（例如 v3）。預設：v3
- N_ENVS：並行環境數量。預設：8
- TOTAL_TIMESTEPS：總訓練步數。預設：1000000
- OPEN_MARIO_DEVICE：'cuda' 或 'cpu'（選用，若非 'cuda' 則預設使用 cpu）

範例（PowerShell）

- 使用預設（world 1 stage 1）：

    python .\mario.py

- 指定 world 2 stage 3、使用 GPU 並 4 個環境：

    $env:MARIO_WORLD='2'; $env:MARIO_STAGE='3'; $env:N_ENVS='4'; $env:OPEN_MARIO_DEVICE='cuda'; python .\mario.py

- 更換模型與訓練步數：

    $env:MARIO_MODEL='PPO'; $env:TOTAL_TIMESTEPS='500000'; python .\mario.py

注意事項

- `requirements.txt` 列出大部分 Python 相依套件。請先依照上方步驟安裝正確的 torch wheel，再使用 `requirements.txt` 安裝其餘套件。
- 若需要每個 worker 玩不同關卡（multi-level）或想加入 CLI（argparse），請修改 `mario.py` 或提出需求，我可以用最小變更協助加入。

授權

此專案目前未提供授權檔；若為私人實驗可視為私人專案，若要公開請新增適當授權條款。
