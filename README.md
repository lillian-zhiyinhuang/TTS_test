
# GPT-SoVITS 語音轉換與 TTS 系統

這是一個功能強大的少樣本語音轉換與文字轉語音（Text-to-Speech, TTS）專案，提供完整的 Web 使用者介面，旨在讓使用者僅需少量樣本就能複製並生成特定音色的語音。

---

## 📦 原始專案與修改版本

- **原始專案**: [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- **修改版本**: [lillian-zhiyinhuang/TTS_test](https://github.com/lillian-zhiyinhuang/TTS_test)

### 🔧 修改內容（相對於原始專案）

```

GPT-SoVITS/
├── Colab-WebUI.ipynb          # ✅ 有修改
├── webui.py                   # ✅ 有修改
├── dataset/                   # ✅ 新增
│   ├── audios/               # ✅ 新增
│   └── metadata.list         # ✅ 新增
├── GPT_SoVITS/
│   ├── inference_webui.py    # ✅ 有修改
│   ├── s1_train.py           # ✅ 有修改
│   ├── s2_train.py           # ✅ 有修改
│   ├── AR/
│   │   ├── data/
│   │   │   ├── data_module.py         # ✅ 有修改
│   │   │   └── dataset.py             # ✅ 有修改
│   │   ├── models/
│   │   │   ├── t2s_lightning_module.py # ✅ 有修改
│   │   │   └── t2s_model.py           # ✅ 有修改
│   ├── module/data_utils.py          # ✅ 有修改
│   ├── prepare_datasets/
│   │   ├── 1-get-text.py             # ✅ 有修改
│   │   └── 3-get-semantic.py         # ✅ 有修改
│   ├── text/
│   │   ├── cleaner.py                # ✅ 有修改
│   │   ├── symbols.py                # ✅ 有修改
│   │   ├── symbols2.py               # ✅ 有修改
│   │   ├── taiwanese_symbols.py     # ✅ 新增
│   │   └── taiwanese.py             # ✅ 新增

```

---

## 🚀 專案使用方式

### WebUI（已棄用）
- 整合包用戶: 執行 `go-webui.bat` 或 `go-webui.ps1`
- 手動安裝用戶: 專案根目錄下執行 `python webui.py`
- Colab 使用者: 執行 `Colab-WebUI.ipynb`

---

## 🧪 推薦流程：Docker 或臺智雲容器訓練

本專案已提供 **Docker 支援** 與 **臺智雲 CCS 容器建構流程**。

### 📌 Docker 建置步驟

```bash
# 建立 Docker 映像檔（Lite 模式，無 ASR、UVR 模型）
docker build --build-arg CUDA_VERSION=12.6 --build-arg LITE=false -t gpt-sovits-local-env .

# 啟動容器
sudo docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  --name gpt-sovits \
  gpt-sovits-local-env
```

如需使用完整功能（包含 UVR、ASR），請改用 `LITE=false` 建置完整環境，或改為臺智雲方案。

### 📌 臺智雲 CCS 容器建置指南

請參考 [使用開發型容器在臺智雲端進行 GPT-SoVITS 訓練指南](TWCC.md)，內含完整指令與路徑設定。

---

## 🔧 訓練流程步驟

### 0. 資料集準備

1. 人聲伴奏分離（UVR5）
2. 語音切分
3. 語音辨識（ASR）
4. 文本校對與標注 `.list` 檔案

### 1. 數據預處理

```bash
python prepare_datasets/1-get-text.py
python prepare_datasets/2-get-hubert-wav32k.py
python prepare_datasets/3-get-semantic.py
```

### 2. 模型訓練

```bash
python s2_train.py --config my_s2_config.json
python s1_train.py --config_file my_s1_config.yaml
```

### 3. 推理生成

* 提供參考音訊與文字後，即可合成語音。
* 支援不同語言、語者風格、情緒建模。

---

## 📁 資料結構與標註格式

```bash
dataset/
├── audios/
│   ├── audio_1.wav
│   └── audio_2.wav
└── metadata.list
```

`metadata.list` 格式：

```
audios/audio_1.wav|speaker1|tw|你好世界。
```

* `tw`: 表示台語音色，支援台羅數字調

---

## 🔌 API 說明

### V1: `api.py`

* `/`：TTS 推理（GET/POST）
* `/change_refer`：變更參考音訊
* `/control`：控制重啟/退出

### V2: `api_v2.py`

* `/tts`：TTS 推理（多參考、參數化）
* `/control`：控制重啟/退出
* `/set_gpt_weights` & `/set_sovits_weights`：模型熱切換

---

## 🧱 專案依賴與環境建置

### 1. 系統需求

* **Python**: 3.10 \~ 3.12
* **PyTorch**: >= 2.2.2（需與 CUDA 相容）
* **GPU**: NVIDIA CUDA >= 11.8
* **macOS**: 支援 MPS，但品質較差建議使用 CPU
* **工具**: FFmpeg, CMake

### 2. 套件依賴（requirements.txt）

* torch, torchaudio, pytorch-lightning, gradio
* numpy, scipy, transformers, onnxruntime-gpu
* funasr, faster-whisper, sentencepiece, jieba\_fast
* fastapi, uvicorn

---

## 🧩 專案待解問題

1. ❓ **模型訓練效果可再評估**
2. ⚠️ **音訊生成耗時較久**
3. 📚 **中文語料斷詞與發音標註**仍需改進

目前使用數字調台羅提供音素學習，亦有探索 LLM 輔助轉換器以強化準備文本。

---

## 📚 延伸閱讀與參考資源

* [原始作者開源頁面](https://github.com/RVC-Boss/GPT-SoVITS)

---

