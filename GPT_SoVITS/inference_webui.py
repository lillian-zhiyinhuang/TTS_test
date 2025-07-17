# /GPT_SoVITS/inference_webui.py (最終整合修正版)

import os
import sys
import traceback
import gradio as gr
import logging

# --- 核心修正 1：確保路徑正確 ---
now_dir = os.getcwd()
sys.path.append(now_dir)
# --- 修正結束 ---

import warnings
warnings.filterwarnings("ignore")
import json
import re
import torch
import librosa
import numpy as np
import torchaudio
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from config import change_choices, get_weights_names, name2gpt_path, name2sovits_path, pretrained_sovits_name
from feature_extractor import cnhubert
from module.models import Generator, SynthesizerTrn, SynthesizerTrnV3
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from tools.i18n.i18n import I18nAuto, scan_language_list
from time import time as ttime
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text.LangSegmenter import LangSegmenter
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from tools.assets import css, js, top_html

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)


# --- 核心修正 2：預先載入詞彙表以進行精準診斷 ---
from text.symbols2 import symbols
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
# --- 修正結束 ---


language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

version = model_version = os.environ.get("version", "v2")


SoVITS_names, GPT_names = get_weights_names()

path_sovits_v3 = pretrained_sovits_name["v3"]
path_sovits_v4 = pretrained_sovits_name["v4"]
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)

if os.path.exists("./weight.json"):
    pass
else:
    with open("./weight.json", "w", encoding="utf-8") as file:
        json.dump({"GPT": {}, "SoVITS": {}}, file)

with open("./weight.json", "r", encoding="utf-8") as file:
    weight_data = file.read()
    weight_data = json.loads(weight_data)
    gpt_path = os.environ.get("gpt_path", weight_data.get("GPT", {}).get(version, GPT_names[-1]))
    sovits_path = os.environ.get("sovits_path", weight_data.get("SoVITS", {}).get(version, SoVITS_names[0]))
    if isinstance(gpt_path, list):
        gpt_path = gpt_path[0]
    if isinstance(sovits_path, list):
        sovits_path = sovits_path[0]

cnhubert_base_path = os.environ.get("cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
bert_path = os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
is_half = False
print(f"推理模式已強制設定為全精度 (FP32) 以確保最高穩定性。 is_half={is_half}")
punctuation = set(["!", "?", "…", ",", ".", "-", " "])


cnhubert.cnhubert_base_path = cnhubert_base_path

import random

from GPT_SoVITS.module.models import Generator, SynthesizerTrn, SynthesizerTrnV3


def set_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000000)
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

from time import time as ttime

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from peft import LoraConfig, get_peft_model
from text import cleaned_text_to_sequence
from text.cleaner import clean_text

from tools.assets import css, js, top_html
from tools.i18n.i18n import I18nAuto, scan_language_list

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dict_language_v1 = {
    i18n("中文"): "all_zh", "English": "en", i18n("日文"): "all_ja",
    i18n("中英混合"): "zh", i18n("日英混合"): "ja", i18n("多语种混合"): "auto",
    i18n("台語"): "tw",
}
dict_language_v2 = {
    i18n("中文"): "all_zh", "English": "en", i18n("日文"): "all_ja",
    i18n("粤语"): "all_yue", i18n("韩文"): "all_ko", i18n("中英混合"): "zh",
    i18n("日英混合"): "ja", i18n("粤英混合"): "yue", i18n("韩英混合"): "ko",
    i18n("多语种混合"): "auto", i18n("多语种混合(粤语)"): "auto_yue",
    i18n("台語"): "tw",
}
dict_language = dict_language_v1 if version == "v1" else dict_language_v2

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

v3v4set = {"v3", "v4"}


def change_sovits_weights(sovits_path, prompt_language=None, text_language=None):
    if "！" in sovits_path or "!" in sovits_path:
        sovits_path = name2sovits_path[sovits_path]
    global vq_model, hps, version, model_version, dict_language, if_lora_v3
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    print(sovits_path, version, model_version, if_lora_v3)
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
    if if_lora_v3 == True and is_exist == False:
        info = path_sovits + "SoVITS %s" % model_version + i18n("底模缺失，无法加载相应 LoRA 权重")
        gr.Warning(info)
        raise FileExistsError(info)
    dict_language = dict_language_v1 if version == "v1" else dict_language_v2
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = (
                {"__type__": "update"},
                {"__type__": "update", "value": prompt_language},
            )
        else:
            prompt_text_update = {"__type__": "update", "value": ""}
            prompt_language_update = {"__type__": "update", "value": i18n("中文")}
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {"__type__": "update"}, {"__type__": "update", "value": text_language}
        else:
            text_update = {"__type__": "update", "value": ""}
            text_language_update = {"__type__": "update", "value": i18n("中文")}
        if model_version in v3v4set:
            visible_sample_steps = True
            visible_inp_refs = False
        else:
            visible_sample_steps = False
            visible_inp_refs = True
        yield (
            {"__type__": "update", "choices": list(dict_language.keys())},
            {"__type__": "update", "choices": list(dict_language.keys())},
            prompt_text_update,
            prompt_language_update,
            text_update,
            text_language_update,
            {
                "__type__": "update",
                "visible": visible_sample_steps,
                "value": 32 if model_version == "v3" else 8,
                "choices": [4, 8, 16, 32, 64, 128] if model_version == "v3" else [4, 8, 16, 32],
            },
            {"__type__": "update", "visible": visible_inp_refs},
            {"__type__": "update", "value": False, "interactive": True if model_version not in v3v4set else False},
            {"__type__": "update", "visible": True if model_version == "v3" else False},
            {"__type__": "update", "value": i18n("模型加载中，请等待"), "interactive": False},
        )

    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version
    if model_version not in v3v4set:
        if "Pro" not in model_version:
            model_version = version
        else:
            hps.model.version = model_version
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    else:
        hps.model.version = model_version
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    if "pretrained" not in sovits_path:
        try:
            del vq_model.enc_q
        except:
            pass
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    if if_lora_v3 == False:
        print("loading sovits_%s" % model_version, vq_model.load_state_dict(dict_s2["weight"], strict=False))
    else:
        path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
        print(
            "loading sovits_%spretrained_G" % model_version,
            vq_model.load_state_dict(load_sovits_new(path_sovits)["weight"], strict=False),
        )
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        print("loading sovits_%s_lora%s" % (model_version, lora_rank))
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        vq_model.eval()

    yield (
        {"__type__": "update", "choices": list(dict_language.keys())},
        {"__type__": "update", "choices": list(dict_language.keys())},
        prompt_text_update,
        prompt_language_update,
        text_update,
        text_language_update,
        {
            "__type__": "update",
            "visible": visible_sample_steps,
            "value": 32 if model_version == "v3" else 8,
            "choices": [4, 8, 16, 32, 64, 128] if model_version == "v3" else [4, 8, 16, 32],
        },
        {"__type__": "update", "visible": visible_inp_refs},
        {"__type__": "update", "value": False, "interactive": True if model_version not in v3v4set else False},
        {"__type__": "update", "visible": True if model_version == "v3" else False},
        {"__type__": "update", "value": i18n("合成语音"), "interactive": True},
    )
    with open("./weight.json") as f:
        data = f.read()
        data = json.loads(data)
        data["SoVITS"][version] = sovits_path
    with open("./weight.json", "w") as f:
        f.write(json.dumps(data))


try:
    next(change_sovits_weights(sovits_path))
except:
    pass


def change_gpt_weights(gpt_path):
    if "！" in gpt_path or "!" in gpt_path:
        gpt_path = name2gpt_path[gpt_path]
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    with open("./weight.json") as f:
        data = f.read()
        data = json.loads(data)
        data["GPT"][version] = gpt_path
    with open("./weight.json", "w") as f:
        f.write(json.dumps(data))
# /GPT_SoVITS/inference_webui.py (請只替換 change_gpt_weights 函數)

# def change_gpt_weights(gpt_path):
#     if "！" in gpt_path or "!" in gpt_path:
#         gpt_path = name2gpt_path[gpt_path]
    
#     # --- 【核心修正】強制重載詞彙表，確保與新選的GPT模型匹配 ---
#     print("--- Sunny's Fix: Forcing vocabulary module reload and rebuild ---")
#     try:
#         # 步驟 1: 導入 importlib 模組，用於重載
#         import importlib
#         # 步驟 2: 導入您的主 symbols 模組 (根據您的專案結構，可能是 symbols 或 symbols2)
#         from text import symbols as symbols_module
        
#         # 步驟 3: 強制 Python 重新從硬碟讀取這個模組檔案，拋棄記憶體快取
#         importlib.reload(symbols_module)
        
#         # 步驟 4: 從新鮮加載的模組中，獲取最新的 symbols 列表
#         symbols = symbols_module.symbols
        
#         # 步驟 5: 重新生成一個全新的、正確的 _symbol_to_id 全域映射表
#         global _symbol_to_id
#         _symbol_to_id = {s: i for i, s in enumerate(symbols)}
        
#         print(f"✅ GPT模型切換時，詞彙表已強制刷新，總符號數: {len(symbols)}")

#     except Exception as e:
#         print(f"🔴 強制刷新詞彙表失敗，請檢查代碼: {e}")
#     # --- 修正結束 ---

#     global hz, max_sec, t2s_model, config
#     hz = 50
#     dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
#     config = dict_s1["config"]
#     max_sec = config["data"]["max_sec"]
#     t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
#     t2s_model.load_state_dict(dict_s1["weight"])
#     if is_half == True:
#         t2s_model = t2s_model.half()
#     t2s_model = t2s_model.to(device)
#     t2s_model.eval()
    
#     with open("./weight.json") as f:
#         data = f.read()
#         data = json.loads(data)
#         data["GPT"][version] = gpt_path
#     with open("./weight.json", "w") as f:
#         f.write(json.dumps(data))


change_gpt_weights(gpt_path)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch

now_dir = os.getcwd()


def clean_hifigan_model():
    global hifigan_model
    if hifigan_model:
        hifigan_model = hifigan_model.cpu()
        hifigan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass


def clean_bigvgan_model():
    global bigvgan_model
    if bigvgan_model:
        bigvgan_model = bigvgan_model.cpu()
        bigvgan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass


def clean_sv_cn_model():
    global sv_cn_model
    if sv_cn_model:
        sv_cn_model.embedding_model = sv_cn_model.embedding_model.cpu()
        sv_cn_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass


def init_bigvgan():
    global bigvgan_model, hifigan_model, sv_cn_model
    from BigVGAN import bigvgan

    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
        use_cuda_kernel=False,
    )
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    clean_hifigan_model()
    clean_sv_cn_model()
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)


def init_hifigan():
    global hifigan_model, bigvgan_model, sv_cn_model
    hifigan_model = Generator(
        initial_channel=100,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        gin_channels=0,
        is_bias=True,
    )
    hifigan_model.eval()
    hifigan_model.remove_weight_norm()
    state_dict_g = torch.load(
        "%s/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth" % (now_dir,),
        map_location="cpu",
        weights_only=False,
    )
    print("loading vocoder", hifigan_model.load_state_dict(state_dict_g))
    clean_bigvgan_model()
    clean_sv_cn_model()
    if is_half == True:
        hifigan_model = hifigan_model.half().to(device)
    else:
        hifigan_model = hifigan_model.to(device)


from sv import SV


def init_sv_cn():
    global hifigan_model, bigvgan_model, sv_cn_model
    sv_cn_model = SV(device, is_half)
    clean_bigvgan_model()
    clean_hifigan_model()


bigvgan_model = hifigan_model = sv_cn_model = None
if model_version == "v3":
    init_bigvgan()
if model_version == "v4":
    init_hifigan()
if model_version in {"v2Pro", "v2ProPlus"}:
    init_sv_cn()

resample_transform_dict = {}


def resample(audio_tensor, sr0, sr1, device):
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)


def get_spepc(hps, filename, dtype, device, is_v2pro=False):
    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load(filename)
    if sr0 != sr1:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        audio = resample(audio, sr0, sr1, device)
    else:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)

    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = spec.to(dtype)
    if is_v2pro == True:
        audio = resample(audio, sr1, 16000, device).to(dtype)
    return spec, audio


# --- 核心修正 3：帶有詞彙表診斷功能的文本清理函數 ---
def clean_text_inf(text: str, language: str, version: str):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    
    # 【關鍵診斷步驟】校驗每一個音素是否存在於詞彙表中
    for p in phones:
        if p not in _symbol_to_id:
            raise ValueError(
                f"\n\n=========================== 致命錯誤：發現未知音素 ===========================\n"
                f"音素 '{p}' (在文本 '{norm_text}' 中) 不存在於您的模型詞彙表中。\n"
                f"請檢查您的 fine-tuning 詞彙表 (text/symbols.py) 與台語文本清理器 (text/cleaner.py)。\n"
                f"=================================================================================\n"
            )
            
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text
# --- 修正結束 ---


dtype = torch.float16 if is_half == True else torch.float32


def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    # 此處不再需要 if language == "tw" 的判斷，因為調用此函數的地方已確保只在 lang=="zh" 時調用
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half else torch.float32,
        ).to(device)
    return bert


splits = {
    "，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…",
}


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


# --- 核心修正 4：恢復 'tw' 獨立性，不再錯誤地映射到 'zh' ---
def get_phones_and_bert(text: str, language: str):
    text = re.sub(r' {2,}', ' ', text).strip()
    
    # 1. 語言切分，忠實反映用戶選擇
    if language.startswith("all_"):
        lang = language.replace("all_", "")
        textlist = [text]
        langlist = [lang]
    elif language == "tw": # 明確處理台語
        textlist = [text]
        langlist = ["tw"]
    else: # 其他混合語言模式
        textlist, langlist = LangSegmenter.getTexts(text)

    print("切分后的文本片段:", textlist)
    print("切分后的語言:", langlist)

    # 2. 文本清理與特徵提取
    phones_list, bert_list, norm_text_list = [], [], []
    for i in range(len(textlist)):
        lang = langlist[i]
        
        # 【重要】調用帶有診斷功能的 clean_text_inf
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
        
        bert = None
        # 【重要】只有在語言確實是 'zh' 時才生成 BERT 特徵
        if lang == "zh" and len(norm_text) > 0:
            bert = get_bert_inf(phones, word2ph, norm_text, lang)

        phones_list.append(phones)
        norm_text_list.append(norm_text)
        if bert is not None:
            bert_list.append(bert)
    
    # 3. 結果匯總
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)
    
    bert = None
    if bert_list:
        bert = torch.cat(bert_list, dim=1)

    return phones, bert, norm_text
# --- 修正結束 ---


from module.mel_processing import mel_spectrogram_torch, spectrogram_torch

spec_min = -12
spec_max = 2


def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


mel_fn = lambda x: mel_spectrogram_torch(x, **{"n_fft": 1024, "win_size": 1024, "hop_size": 256, "num_mels": 100, "sampling_rate": 24000, "fmin": 0, "fmax": None, "center": False})
mel_fn_v4 = lambda x: mel_spectrogram_torch(x, **{"n_fft": 1280, "win_size": 1280, "hop_size": 320, "num_mels": 100, "sampling_rate": 32000, "fmin": 0, "fmax": None, "center": False})


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


sr_model = None


def audio_sr(audio, sr):
    global sr_model
    if sr_model == None:
        from tools.audio_sr import AP_BWE

        try:
            sr_model = AP_BWE(device, DictToAttrRecursive)
        except FileNotFoundError:
            gr.Warning(i18n("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载好"))
            return audio.cpu().detach().numpy(), sr
    return sr_model(audio, sr)


cache = {}


# --- 核心修正 5：確保 get_tts_wav 能處理 bert 為 None 的情況 ---
# /GPT_SoVITS/inference_webui.py (請只替換 get_tts_wav 函數)

# def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free, speed, if_freeze, inp_refs, sample_steps, if_sr, pause_second):
#     if not ref_wav_path: gr.Warning("請上傳參考音訊！"); return None, None
#     if not text: gr.Warning("請填入目標合成文本！"); return None, None

#     # --- 決定性診斷：獲取模型真實的詞彙表大小 ---
#     # t2s_model 是全局變量，此時已經加載
#     # 修正後的程式碼：
#     max_phoneme_id = t2s_model.model.ar_text_embedding.word_embeddings.num_embeddings
#     # ---------------------------------------------

#     prompt_language = dict_language[prompt_language]
#     text_language = dict_language[text_language]
#     if not ref_free: prompt_text = prompt_text.strip("\n")
#     text = text.strip("\n")
    
#     with torch.no_grad():
#         wav16k, _ = librosa.load(ref_wav_path, sr=16000)
#         wav16k = torch.from_numpy(wav16k).to(device)
#         if is_half: wav16k = wav16k.half()
#         ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
#         codes = vq_model.extract_latent(ssl_content)
#         prompt_semantic = codes[0, 0]

#     texts = text.split("\n")
#     audio_opt = []
    
#     phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language) if not ref_free else (None, None, None)

#     for text_segment in texts:
#         if not text_segment.strip(): continue
#         if text_segment[-1] not in splits: text_segment += "。" if text_language != "en" else "."
        
#         phones2, bert2, norm_text2 = get_phones_and_bert(text_segment, text_language)
        
#         bert = None
#         if not ref_free:
#             phones = phones1 + phones2
#             if bert1 is not None and bert2 is not None:
#                 bert = torch.cat([bert1, bert2], 1)
#             elif bert1 is not None:
#                 bert = bert1
#             elif bert2 is not None:
#                 bert = bert2
#         else:
#             phones = phones2
#             bert = bert2
        
#         # --- 決定性診斷：在送入模型前，檢查所有 ID 是否越界 ---
#         for p_id in phones:
#             if p_id >= max_phoneme_id:
#                 # 找到問題ID，拋出包含所有關鍵資訊的錯誤！
#                 problem_symbol = list(_symbol_to_id.keys())[list(_symbol_to_id.values()).index(p_id)]
#                 raise ValueError(
#                     f"\n\n=========================== 致命錯誤：詞彙表 ID 越界 ===========================\n"
#                     f"文本 '{norm_text1 if norm_text1 else ''}{norm_text2}' 中的音素 '{problem_symbol}' 被轉換為 ID: {p_id}\n"
#                     f"但是，您加載的 GPT 模型在訓練時，其詞彙表大小僅為: {max_phoneme_id} (最大合法 ID 為 {max_phoneme_id - 1})。\n"
#                     f"原因：當前環境的 symbols.py 比模型訓練時的要新、要大。\n"
#                     f"解決方案：\n"
#                     f"1. (推薦) 找到訓練時使用的舊版 symbols.py，並在當前環境中替換它。\n"
#                     f"2. (備選) 使用當前較新的 symbols.py 重新對您的模型進行 fine-tune。\n"
#                     f"=================================================================================\n"
#                 )
#         # --- 診斷結束 ---

#         all_phoneme_ids = torch.LongTensor(phones).to(device).unsqueeze(0)
#         all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]], device=device)
#         prompt = prompt_semantic.unsqueeze(0) if not ref_free else None

#         with torch.no_grad():
#             pred_semantic, idx = t2s_model.model.infer_panel(
#                 all_phoneme_ids, all_phoneme_len, prompt, bert,
#                 top_k=top_k, top_p=top_p, temperature=temperature,
#                 early_stop_num=hz * max_sec
#             )
        
#         pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
#         refer, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device)
#         refer = refer.to(device)
#         if is_half: refer = refer.half()
        
#         audio = vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer)[0, 0].data.cpu().float().numpy()
#         audio_opt.append(audio)
#         audio_opt.append(np.zeros(int(hps.data.sampling_rate * pause_second), dtype=np.float32))

#     yield hps.data.sampling_rate, np.concatenate(audio_opt)

# /GPT_SoVITS/inference_webui.py (請只替換 get_tts_wav 函數)

# def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free, speed, if_freeze, inp_refs, sample_steps, if_sr, pause_second):
#     # ------------------- Sunny's Final Diagnostic & Fix -------------------
#     if not ref_wav_path:
#         gr.Warning("請上傳參考音訊！")
#         return None, None
#     if not text:
#         gr.Warning("請填入目標合成文本！")
#         return None, None

#     # 【決定性診斷】: 獲取模型真實的詞彙表大小
#     # t2s_model 是全局變量，此時已經加載
#     try:
#         # v2Pro 模型的 embedding layer 路徑
#         actual_model_vocab_size = t2s_model.model.ar_text_embedding.word_embeddings.num_embeddings
#         env_vocab_size = len(symbols)
#         print(f"✅ 診斷資訊: 已加載的 GPT 模型內部詞彙表大小為: {actual_model_vocab_size}")
#         print(f"   而當前 WebUI 環境的 `symbols` 列表大小為: {env_vocab_size}")
#         if actual_model_vocab_size != env_vocab_size:
#             print(f"   🔴 警告：維度不匹配！模型與當前環境的詞彙表大小不一致。")
#     except Exception as e:
#         print(f"🔴 獲取模型詞彙表大小時出錯: {e}")
#         actual_model_vocab_size = 9999 # 設置一個預設大數以繼續

#     prompt_language = dict_language[prompt_language]
#     text_language = dict_language[text_language]
#     if not ref_free: prompt_text = prompt_text.strip("\n")
#     text = text.strip("\n")
    
#     with torch.no_grad():
#         wav16k, _ = librosa.load(ref_wav_path, sr=16000)
#         wav16k = torch.from_numpy(wav16k).to(device)
#         if is_half: wav16k = wav16k.half()
#         ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
#         codes = vq_model.extract_latent(ssl_content)
#         prompt_semantic = codes[0, 0]

#     texts = text.split("\n")
#     audio_opt = []
    
#     phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language) if not ref_free else (None, None, None)

#     for text_segment in texts:
#         if not text_segment.strip(): continue
#         if text_segment[-1] not in splits: text_segment += "。" if text_language != "en" else "."
        
#         phones2, bert2, norm_text2 = get_phones_and_bert(text_segment, text_language)
        
#         bert = None
#         if not ref_free:
#             phones = phones1 + phones2
#             if bert1 is not None and bert2 is not None: bert = torch.cat([bert1, bert2], 1)
#             elif bert1 is not None: bert = bert1
#             elif bert2 is not None: bert = bert2
#         else:
#             phones = phones2
#             bert = bert2
        
#         # 【全新的防錯機制】: 在送入模型前，檢查所有 ID 是否越界
#         for p_id in phones:
#             if p_id >= actual_model_vocab_size:
#                 # 找到問題ID，拋出包含所有關鍵資訊的錯誤！
#                 problem_symbol = list(_symbol_to_id.keys())[list(_symbol_to_id.values()).index(p_id)]
#                 raise ValueError(
#                     f"\n\n=========================== 致命錯誤：詞彙表 ID 越界 ===========================\n"
#                     f"文本 '{norm_text1 if norm_text1 else ''}{norm_text2}' 中的音素 '{problem_symbol}' 被轉換為 ID: {p_id}。\n"
#                     f"但是，您加載的 GPT 模型在訓練時，其詞彙表大小僅為: {actual_model_vocab_size} (最大合法 ID 為 {actual_model_vocab_size - 1})。\n"
#                     f"原因：您當前環境的 `symbols` 列表比模型訓練時所用的要【小】或【順序不一致】。\n"
#                     f"解決方案：\n"
#                     f"1. (推薦) 找到您訓練此模型時所用的原始 `symbols.py` 檔案，並用它替換當前環境中的檔案，然後重啟 WebUI。\n"
#                     f"2. (備選) 如果找不到原始檔案，您必須使用當前這個 `symbols.py` 檔案【重新訓練】一個新模型。\n"
#                     f"=================================================================================\n"
#                 )

#         all_phoneme_ids = torch.LongTensor(phones).to(device).unsqueeze(0)
#         all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]], device=device)
#         prompt = prompt_semantic.unsqueeze(0) if not ref_free else None

#         with torch.no_grad():
#             pred_semantic, idx = t2s_model.model.infer_panel(
#                 all_phoneme_ids, all_phoneme_len, prompt, bert,
#                 top_k=top_k, top_p=top_p, temperature=temperature,
#                 early_stop_num=hz * max_sec
#             )
        
#         # 後續程式碼與原版相同...
#         pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
#         refer, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device)
#         refer = refer.to(device)
#         if is_half: refer = refer.half()
        
#         # audio = vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer)[0, 0].data.cpu().float().numpy()
#         # ---------------------------------- 0717
#         # 【最終修正】使用上面一行程式碼已經為我們準備好的 `audio_tensor`
#         # 而不是傳遞檔案路徑字串 `ref_wav_path`
#         speaker_embedding = sv_cn_model.compute_embedding3(audio_tensor)

#         # 將正確提取的 speaker_embedding 傳遞給 decode 函式
#         audio = vq_model.decode(
#             pred_semantic,
#             torch.LongTensor(phones2).to(device).unsqueeze(0),
#             refer,
#             speaker_embedding  # 現在這裡傳入的是正確的「聲音身份證」
#         )[0, 0].data.cpu().float().numpy()
#         # ----------------------------
#         audio_opt.append(audio)
#         audio_opt.append(np.zeros(int(hps.data.sampling_rate * pause_second), dtype=np.float32))

#     # yield hps.data.sampling_rate, np.concatenate(audio_opt) # 原版輸出
#     # 為了讓 Gradio 能顯示我們拋出的詳細錯誤，需要 try-except
#     try:
#         # 這裡的 yield 只是為了觸發上面的邏輯
#         final_audio = np.concatenate(audio_opt)
#         yield hps.data.sampling_rate, final_audio
#     except ValueError as e:
#         # 將我們客製化的 ValueError 顯示在 Gradio 的錯誤提示中
#         raise gr.Error(str(e))
#     # ------------------- Sunny's Final Diagnostic & Fix Ends -------------------


def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free, speed, if_freeze, inp_refs, sample_steps, if_sr, pause_second):
    
    # 預先檢查，避免不必要的運算
    if not ref_wav_path:
        gr.Warning("請上傳參考音訊！")
        return None, None
    if not text:
        gr.Warning("請填入目標合成文本！")
        return None, None

    # 將語言選項轉為內部代號
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    # 清理輸入文本
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
    text = text.strip("\n")
    
    # 開始合成流程
    with torch.no_grad():
        # 1. 提取參考音訊的語義特徵 (Prompt)
        wav16k, _ = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        if is_half:
            wav16k = wav16k.half()
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]

    # 將長文本切分成多個句子
    texts = text.split("\n")
    audio_opt = []
    
    # 提取參考文本的音素與BERT特徵（如果需要）
    phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language) if not ref_free else (None, None, None)

    # 逐句進行合成
    for text_segment in texts:
        if not text_segment.strip():
            continue
        if text_segment[-1] not in splits:
            text_segment += "。" if text_language != "en" else "."
        
        # 提取目標文本的音素與BERT特徵
        phones2, bert2, norm_text2 = get_phones_and_bert(text_segment, text_language)
        
        # 根據是否使用參考文本，合併音素和BERT特徵
        if not ref_free:
            phones = phones1 + phones2
            bert = torch.cat([bert1, bert2], 1) if bert1 is not None and bert2 is not None else (bert1 if bert1 is not None else bert2)
        else:
            phones = phones2
            bert = bert2
        
        # 準備送入GPT模型的最終資料
        all_phoneme_ids = torch.LongTensor(phones).to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]], device=device)
        prompt = prompt_semantic.unsqueeze(0) if not ref_free else None

        # 2. GPT模型：從音素生成語義Token
        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids, all_phoneme_len, prompt, bert,
                top_k=top_k, top_p=top_p, temperature=temperature,
                early_stop_num=hz * max_sec
            )
        
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        
        # 3. So-VITS模型：從語義Token生成音訊
        
        # 準備So-VITS所需的參考頻譜圖 和 【聲音身份證】
        refer, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device, is_v2pro=True)
        
        # 【最終修正】確保 speaker_embedding 在 v2Pro 模型中被正確提取和傳遞
        speaker_embedding = None
        if model_version in {"v2Pro", "v2ProPlus"}:
            if sv_cn_model is None: # 確保提取器已初始化
                init_sv_cn()
            # 使用 audio_tensor 提取特徵，這是最穩定的方式
            speaker_embedding = sv_cn_model.compute_embedding3(audio_tensor)

        # 呼叫 decode 函式，傳入所有必要的參數
        audio = vq_model.decode(
            pred_semantic,
            torch.LongTensor(phones2).to(device).unsqueeze(0),
            refer.to(device),
            sv_emb=speaker_embedding # 使用 sv_emb 關鍵字參數傳遞
        )[0, 0].data.cpu().float().numpy()

        audio_opt.append(audio)
        audio_opt.append(np.zeros(int(hps.data.sampling_rate * pause_second), dtype=np.float32))

    yield hps.data.sampling_rate, np.concatenate(audio_opt)


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts) > 1 and len(opts[-1]) < 50:
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut4(inp):
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut5(inp):
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def custom_sort_key(s):
    parts = re.split("(\d+)", s)
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError(i18n("请输入有效文本"))
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text


def html_center(text, label="p"):
    return f"""<div style="text-align: center; margin: 100; padding: 50;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


def html_left(text, label="p"):
    return f"""<div style="text-align: left; margin: 0; padding: 0;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, js=js, css=css) as app:
    gr.HTML(
        top_html.format(
            i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.")
            + i18n("如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE.")
        ),
        elem_classes="markdown",
    )
    with gr.Group():
        gr.Markdown(html_center(i18n("模型切换"), "h3"))
        with gr.Row():
            GPT_dropdown = gr.Dropdown(
                label=i18n("GPT模型列表"),
                choices=sorted(GPT_names, key=custom_sort_key),
                value=gpt_path,
                interactive=True,
                scale=14,
            )
            SoVITS_dropdown = gr.Dropdown(
                label=i18n("SoVITS模型列表"),
                choices=sorted(SoVITS_names, key=custom_sort_key),
                value=sovits_path,
                interactive=True,
                scale=14,
            )
            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary", scale=14)
            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
        gr.Markdown(html_center(i18n("*请上传并填写参考信息"), "h3"))
        with gr.Row():
            inp_ref = gr.Audio(label=i18n("请上传3~10秒内参考音频，超过会报错！"), type="filepath", scale=13)
            with gr.Column(scale=13):
                ref_text_free = gr.Checkbox(
                    label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。")
                    + i18n("v3暂不支持该模式，使用了会报错。"),
                    value=False,
                    interactive=True if model_version not in v3v4set else False,
                    show_label=True,
                    scale=1,
                )
                gr.Markdown(
                    html_left(
                        i18n("使用无参考文本模式时建议使用微调的GPT")
                        + "<br>"
                        + i18n("听不清参考音频说的啥(不晓得写啥)可以开。开启后无视填写的参考文本。")
                    )
                )
                prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value="", lines=5, max_lines=5, scale=1)
            with gr.Column(scale=14):
                prompt_language = gr.Dropdown(
                    label=i18n("参考音频的语种"),
                    choices=list(dict_language.keys()),
                    value=i18n("中文"),
                )
                inp_refs = (
                    gr.File(
                        label=i18n(
                            "可选项：通过拖拽多个文件上传多个参考音频（建议同性），平均融合他们的音色。如不填写此项，音色由左側單個參考音訊控制。如是微調模型，建議參考音訊全部在微調訓練集音色內，底模不用管。"
                        ),
                        file_count="multiple",
                    )
                    if model_version not in v3v4set
                    else gr.File(
                        label=i18n(
                            "可选项：通过拖拽多个文件上传多个参考音频（建议同性），平均融合他們的音色。如不填写此项，音色由左側單個參考音訊控制。如是微調模型，建議參考音訊全部在微調訓練集音色內，底模不用管。"
                        ),
                        file_count="multiple",
                        visible=False,
                    )
                )
                sample_steps = (
                    gr.Radio(
                        label=i18n("采样步数,如果觉得电,提高试试,如果觉得慢,降低试试"),
                        value=32 if model_version == "v3" else 8,
                        choices=[4, 8, 16, 32, 64, 128] if model_version == "v3" else [4, 8, 16, 32],
                        visible=True,
                    )
                    if model_version in v3v4set
                    else gr.Radio(
                        label=i18n("采样步数,如果觉得电,提高试试,如果觉得慢,降低试试"),
                        choices=[4, 8, 16, 32, 64, 128] if model_version == "v3" else [4, 8, 16, 32],
                        visible=False,
                        value=32 if model_version == "v3" else 8,
                    )
                )
                if_sr_Checkbox = gr.Checkbox(
                    label=i18n("v3输出如果觉得闷可以试试开超分"),
                    value=False,
                    interactive=True,
                    show_label=True,
                    visible=False if model_version != "v3" else True,
                )
        gr.Markdown(html_center(i18n("*请填写需要合成的目标文本和语种模式"), "h3"))
        with gr.Row():
            with gr.Column(scale=13):
                text = gr.Textbox(label=i18n("需要合成的文本"), value="", lines=26, max_lines=26)
            with gr.Column(scale=7):
                text_language = gr.Dropdown(
                    label=i18n("需要合成的语种") + i18n(".限制范围越小判别效果越好。"),
                    choices=list(dict_language.keys()),
                    value=i18n("中文"),
                    scale=1,
                )
                how_to_cut = gr.Dropdown(
                    label=i18n("怎么切"),
                    choices=[
                        i18n("不切"),
                        i18n("凑四句一切"),
                        i18n("凑50字一切"),
                        i18n("按中文句号。切"),
                        i18n("按英文句号.切"),
                        i18n("按标点符号切"),
                    ],
                    value=i18n("凑四句一切"),
                    interactive=True,
                    scale=1,
                )
                gr.Markdown(value=html_center(i18n("语速调整，高为更快")))
                if_freeze = gr.Checkbox(
                    label=i18n("是否直接对上次合成结果调整语速和音色。防止随机性。"),
                    value=False,
                    interactive=True,
                    show_label=True,
                    scale=1,
                )
                with gr.Row():
                    speed = gr.Slider(
                        minimum=0.6, maximum=1.65, step=0.05, label=i18n("语速"), value=1, interactive=True, scale=1
                    )
                    pause_second_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        step=0.01,
                        label=i18n("句间停顿秒数"),
                        value=0.3,
                        interactive=True,
                        scale=1,
                    )
                gr.Markdown(html_center(i18n("GPT采样参数(无参考文本时不要太低。不懂就用默认)：")))
                top_k = gr.Slider(
                    minimum=1, maximum=100, step=1, label=i18n("top_k"), value=15, interactive=True, scale=1
                )
                top_p = gr.Slider(
                    minimum=0, maximum=1, step=0.05, label=i18n("top_p"), value=1, interactive=True, scale=1
                )
                temperature = gr.Slider(
                    minimum=0, maximum=1, step=0.05, label=i18n("temperature"), value=1, interactive=True, scale=1
                )

        with gr.Row():
            inference_button = gr.Button(value=i18n("合成语音"), variant="primary", size="lg", scale=25)
            output = gr.Audio(label=i18n("输出的语音"), scale=14)

        inference_button.click(
            get_tts_wav,
            [
                inp_ref,
                prompt_text,
                prompt_language,
                text,
                text_language,
                how_to_cut,
                top_k,
                top_p,
                temperature,
                ref_text_free,
                speed,
                if_freeze,
                inp_refs,
                sample_steps,
                if_sr_Checkbox,
                pause_second_slider,
            ],
            [output],
        )
        SoVITS_dropdown.change(
            change_sovits_weights,
            [SoVITS_dropdown, prompt_language, text_language],
            [
                prompt_language,
                text_language,
                prompt_text,
                prompt_language,
                text,
                text_language,
                sample_steps,
                inp_refs,
                ref_text_free,
                if_sr_Checkbox,
                inference_button,
            ],
        )
        GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])


if __name__ == "__main__":
    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=infer_ttswebui,
    )
