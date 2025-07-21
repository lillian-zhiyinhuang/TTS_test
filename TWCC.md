# 使用開發型容器在臺智雲端進行 GPT-SoVITS 訓練指南

這份指南將引導您使用臺智雲的**開發型容器服務 (CCS)**，從零到一完成 GPT-SoVITS 的環境建構與模型訓練。

## 總覽：從 WebUI 到指令行的核心思路

WebUI 上的每一個按鈕，其背後都是在執行一個特定的 Python 腳本 (`.py`)。我們的目標就是找出這些按鈕對應的腳本，並學會如何透過指令行直接傳遞參數給它們來執行，從而擺脫對圖形介面的依賴。

### 整個流程分為六大步驟：

1. **準備雲端儲存 (CFS) 與上傳資料**
2. **建立與啟動開發型容器 (CCS)**
3. **連線至容器並設定 Poetry 環境**
4. **下載預訓練模型**
5. **透過指令執行數據預處理與模型訓練**
6. **監控與取回成果**

### 步驟一：準備雲端儲存 (CFS) 與上傳資料

在容器化的工作流程中，**外部持久化儲存是所有工作的基礎**，因為容器本身是暫時的。

1. **建立雲端檔案系統 (CFS)**：
    - 登入臺智雲，建立一個 **300 GB** 或以上的雲端檔案系統 (CFS)。這是您存放所有程式碼、資料集和模型的地方。
2. **上傳專案與數據集至 CFS**：
    - **在您的本地 macOS 電腦上**，先 `git clone` 專案。
        
        ```
        git clone https://github.com/RVC-Boss/GPT-SoVITS.git
        
        ```
        
    - **使用臺智雲網頁介面或 `rsync`**，將**完整的 `GPT-SoVITS` 資料夾**和您的**音檔資料集資料夾**上傳到 CFS 的根目錄。
        - **強烈建議**啟動一台最便宜的 CPU 虛擬機掛載 CFS，然後使用 `rsync` 指令上傳，完成後即可刪除該 VM。
        
        ```
        # 語法: rsync -avz --progress /本地/資料夾/路徑 使用者名稱@伺服器IP:/遠端CFS掛載點/
        
        # 上傳專案
        rsync -avz --progress ./GPT-SoVITS/ ubuntu@<您的暫時VM公用IP>:/data/GPT-SoVITS/
        
        # 上傳資料集
        rsync -avz --progress /path/to/your/local/dataset/ ubuntu@<您的暫時VM公用IP>:/data/datasets/
        
        ```
        
    - 完成後，您的 CFS 根目錄下應有 `GPT-SoVITS` 和 `datasets` 兩個資料夾。

### 步驟二：建立與啟動開發型容器 (CCS)

這是取代「建立VM」的步驟。

1. **前往臺智雲容器服務 (CCS) 介面**，建立一個新的**開發型容器**。
2. **填寫容器配置**：
    - **映像檔 (Image)**: 選擇臺智雲官方提供的最新版 **PyTorch 映像檔**，例如 `pytorch/pytorch:2.x.x-cuda12.x-cudnn8-runtime`。這已包含所有底層驅動。
    - **硬體資源**: 選擇您預算內的方案，例如 `c.super` (1 GPU, 4 CPU, 90GB RAM)。
    - **儲存掛載 (Volume Mount)**: **此為最關鍵步驟！** 建立兩個掛載點，將您在 CFS 上的資料夾映射到容器內部：
        - **來源路徑 (CFS):** `/GPT-SoVITS` -> **掛載點 (容器內):** `/workspace/GPT-SoVITS`
        - **來源路徑 (CFS):** `/datasets` -> **掛載點 (容器內):** `/workspace/datasets`
    - **啟動指令 (Command)**: 填入 `sleep infinity`。這能讓容器啟動後不自動退出，保持運行狀態等待您連線。
3. **啟動容器**，並等待其進入「運行中」狀態。

### 步驟三：連線至容器並設定 Poetry 環境

1. **連線到容器**：
    - 在臺智雲的容器管理介面，找到您運作中的容器，複製其提供的 SSH 連線指令。
    - 在您的 macOS 終端機貼上並執行，即可進入容器的 Shell。
2. **安裝與設定 Poetry**：
    - 進入容器後，首先安裝 Poetry：
        
        ```
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="/root/.local/bin:$PATH"
        
        ```
        
    - 進入您掛載的專案目錄：
        
        ```
        cd /workspace/GPT-SoVITS
        
        ```
        
    - **強烈建議**將 Poetry 虛擬環境建立在專案目錄內，這樣它也會被保存在 CFS 上：
        
        ```
        poetry config virtualenvs.in-project true
        
        ```
        
3. **安裝 PyTorch 與專案依賴**：
    - 初始化 Poetry 專案：
        
        ```
        poetry init --no-interaction
        
        ```
        
    - 安裝依賴。由於 PyTorch 已由基礎映像檔提供，我們只需安裝 `requirements.txt` 中的其他套件：
        
        ```
        poetry run pip install -r requirements.txt
        poetry run pip install -r extra-req.txt --no-deps
        
        ```
        
4. **啟用 Poetry 環境**：
    - 為了方便後續操作，直接進入 Poetry 建立的虛擬 shell：
        
        ```
        poetry shell
        
        ```
        
    - 執行後，您會看到命令提示符改變，之後的所有 `python` 指令都會在這個隔離環境中執行。

### 步驟四：下載預訓練模型

*此步驟需在已連線至容器，並處於 Poetry shell 環境中執行。*

1. **進入預訓練模型目錄**：
    
    ```
    # 注意路徑是在 /workspace/ 下
    mkdir -p /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models
    cd /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models
    
    ```
    
2. **下載核心模型** (以 V2 版本為例)：
    
    ```
    # 下載 SoVITS V2 模型
    wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2G2333k.pth
    wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2D2333k.pth
    # 下載 GPT V2 模型
    wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
    
    ```
    
3. 返回專案根目錄：
    
    ```
    cd /workspace/GPT-SoVITS
    
    ```
    

### 步驟五：透過指令執行數據預處理與模型訓練

*此步驟需在 `tmux` 中執行，以防斷線。*

1. **啟動 tmux**：
    
    ```
    tmux new -s training
    
    ```
    

### **數據預處理 (一鍵三連)**

*這些腳本透過環境變數傳遞參數。請注意所有路徑都換成容器內的路徑。*

### 4.1 文本提取 (`1-get-text.py`)

```
# 設定環境變數
export inp_text='/path/to/your/dataset.list' # 這裡仍是您 list 檔案的路徑
export inp_wav_dir='/workspace/datasets/' # **注意**：這是容器內的掛載路徑
export exp_name='your_experiment_name'
export opt_dir='logs/your_experiment_name' # 這會被建立在 /workspace/GPT-SoVITS/logs/ 下
export bert_pretrained_dir='GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'
export is_half='True'
export i_part=0
export all_parts=1
export _CUDA_VISIBLE_DEVICES=0

# 執行腳本
python GPT_SoVITS/prepare_datasets/1-get-text.py

```

### 4.2 Hubert 與特徵提取 (`2-get-hubert-wav32k.py`)

```
# 更新部分環境變數
export cnhubert_base_dir='GPT_SoVITS/pretrained_models/chinese-hubert-base'
export sv_path='GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt'

# 執行腳本
python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
# 如果您使用 v2Pro 系列模型，還需執行
python GPT_SoVITS/prepare_datasets/2-get-sv.py

```

### 4.3 語義 Token 提取 (`3-get-semantic.py`)

```
# 選擇您要使用的 SoVITS 底模版本
export pretrained_s2G='GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth'
export s2config_path='GPT_SoVITS/configs/s2.json'

# 執行腳本
python GPT_SoVITS/prepare_datasets/3-get-semantic.py

```

### **模型訓練**

*這兩個步驟透過傳遞一個設定檔 (config file) 來執行。*

### 4.4 SoVITS 訓練 (`s2_train.py`)

1. 在 `/workspace/GPT-SoVITS` 目錄下創建 `my_s2_config.json` 檔案 (內容同前，路徑會自動基於執行位置，所以無需修改)。
2. 執行訓練指令：
    
    ```
    # v1/v2/v2Pro 系列用 s2_train.py
    python GPT_SoVITS/s2_train.py --config "my_s2_config.json"
    
    ```
    

### 4.5 GPT 訓練 (`s1_train.py`)

1. 在 `/workspace/GPT-SoVITS` 目錄下創建 `my_s1_config.yaml` 檔案 (內容同前，路徑會自動基於執行位置，所以無需修改)。
2. 執行訓練指令：
    
    ```
    # 設定使用的 GPU
    export _CUDA_VISIBLE_DEVICES=0
    export hz=25hz
    python GPT_SoVITS/s1_train.py --config_file "my_s1_config.yaml"
    
    ```
    

### 步驟六：監控與取回成果

1. **監控 GPU**：
    - 在另一個 SSH 視窗連線至**同一個容器**，執行 `nvidia-smi -l 1` 來實時監控。
2. **下載模型**：
    - 訓練完成後，模型會被保存在 CFS 上的 `SoVITS_weights_v2Pro` 和 `GPT_weights_v2Pro` 資料夾中。
    - **您無需 `rsync`**。只需**停止您的 GPU 容器**，然後透過臺智雲的**網頁版檔案總管**直接從您的 CFS 中將訓練好的模型檔案下載到本地 macOS 電腦即可。
3. **停止運算以節費**：
    - 所有工作完成後，**務必回到臺智雲控制台「停止」或「刪除」您的開發型容器**，以停止運算計費！您的所有成果都安全地保存在 CFS 中。