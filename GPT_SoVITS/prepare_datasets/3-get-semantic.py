"""
import os

inp_text = os.environ.get("inp_text")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
opt_dir = os.environ.get("opt_dir")
pretrained_s2G = os.environ.get("pretrained_s2G")
s2config_path = os.environ.get("s2config_path")

if os.path.exists(pretrained_s2G):
    ...
else:
    raise FileNotFoundError(pretrained_s2G)
# version=os.environ.get("version","v2")
size = os.path.getsize(pretrained_s2G)
if size < 82978 * 1024:
    version = "v1"
elif size < 100 * 1024 * 1024:
    version = "v2"
elif size < 103520 * 1024:
    version = "v1"
elif size < 700 * 1024 * 1024:
    version = "v2"
else:
    version = "v3"
import torch

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
import traceback
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging
import utils

if version != "v3":
    from module.models import SynthesizerTrn
else:
    from module.models import SynthesizerTrnV3 as SynthesizerTrn
from tools.my_utils import clean_path

logging.getLogger("numba").setLevel(logging.WARNING)
# from config import pretrained_s2G

# inp_text=sys.argv[1]
# exp_name=sys.argv[2]
# i_part=sys.argv[3]
# all_parts=sys.argv[4]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[5]
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name


hubert_dir = "%s/4-cnhubert" % (opt_dir)
semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
if os.path.exists(semantic_path) == False:
    os.makedirs(opt_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    hps = utils.get_hparams_from_file(s2config_path)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        version=version,
        **hps.model,
    )
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)

    vq_model.eval()

    # utils.load_checkpoint(utils.latest_checkpoint_path(hps.s2_ckpt_dir, "G_*.pth"), vq_model, None, True)
    # utils.load_checkpoint(pretrained_s2G, vq_model, None, True)
    print(
        vq_model.load_state_dict(
            torch.load(pretrained_s2G, map_location="cpu", weights_only=False)["weight"], strict=False
        )
    )

    def name2go(wav_name, lines):
        hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
        if os.path.exists(hubert_path) == False:
            return
        ssl_content = torch.load(hubert_path, map_location="cpu")
        if is_half == True:
            ssl_content = ssl_content.half().to(device)
        else:
            ssl_content = ssl_content.to(device)
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines.append("%s\t%s" % (wav_name, semantic))

    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    lines1 = []
    for line in lines[int(i_part) :: int(all_parts)]:
        # print(line)
        try:
            # wav_name,text=line.split("\t")
            wav_name, spk_name, language, text = line.split("|")
            wav_name = clean_path(wav_name)
            wav_name = os.path.basename(wav_name)
            # name2go(name,lines1)
            name2go(wav_name, lines1)
        except:
            print(line, traceback.format_exc())
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines1))
"""

# /GPT_SoVITS/prepare_datasets/3-get-semantic.py (終極完整修正版)

import os
import sys
import torch
import traceback
import logging
import utils

# --- 核心修正：確保導入路徑正確 ---
now_dir = os.getcwd()
sys.path.append(now_dir)
# --- 修正結束 ---

from module.models import SynthesizerTrn
from tools.my_utils import clean_path

# ... (載入環境變數的程式碼保持不變) ...
inp_text = os.environ.get("inp_text")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
opt_dir = os.environ.get("opt_dir")
pretrained_s2G = os.environ.get("pretrained_s2G")
s2config_path = os.environ.get("s2config_path")
is_half_str = os.environ.get("is_half", "True")

# --- 核心修正：正確解析 is_half ---
is_half = eval(is_half_str) if isinstance(is_half_str, str) else bool(is_half_str)
if not torch.cuda.is_available():
    is_half = False
# --- 修正結束 ---

logging.getLogger("numba").setLevel(logging.WARNING)

hubert_dir = f"{opt_dir}/4-cnhubert"
semantic_path = f"{opt_dir}/6-name2semantic-{i_part}.tsv"

if not os.path.exists(semantic_path):
    os.makedirs(opt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 載入設定檔和模型
    hps = utils.get_hparams_from_file(s2config_path)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    ).to(device)
    
    # --- 核心修正：在載入權重前，將模型轉換為半精度 ---
    if is_half:
        vq_model = vq_model.half()
    # --- 修正結束 ---
    
    # 使用我們之前修正過的、穩健的權重載入邏輯
    try:
        pretrain_dict = torch.load(pretrained_s2G, map_location="cpu")["weight"]
        model_dict = vq_model.state_dict()
        tmp_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(tmp_dict)
        vq_model.load_state_dict(model_dict, strict=False)
        print("✅ 權重載入完成，已成功跳過不匹配的層。")
    except Exception as e:
        print(f"❌ 載入預訓練 SoVITS 模型失敗: {e}")
        
    vq_model.eval()

    def name2go(wav_name, lines_data):
        hubert_path = f"{hubert_dir}/{wav_name}.pt"
        if not os.path.exists(hubert_path):
            return
        
        ssl_content = torch.load(hubert_path, map_location="cpu").to(device)
        
        # --- 核心修正：確保輸入數據與模型精度一致 ---
        if is_half:
            ssl_content = ssl_content.half()
        # --- 修正結束 ---
        
        with torch.no_grad():
            codes = vq_model.extract_latent(ssl_content)
            
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines_data.append(f"{wav_name}\t{semantic}")

    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    processed_lines = []
    for line in lines[int(i_part) :: int(all_parts)]:
        try:
            wav_name, spk_name, language, text = line.split("|")
            wav_name = os.path.basename(clean_path(wav_name))
            name2go(wav_name, processed_lines)
        except:
            print(f"處理行時出錯: {line}")
            traceback.print_exc()
            
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(processed_lines))
    
    print(f"✅ 第 {i_part} 部分語義特徵提取完成。")