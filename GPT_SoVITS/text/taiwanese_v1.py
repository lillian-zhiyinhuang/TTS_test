# taiwanese.py (最終修正版)

import re
from taibun import Converter
import opencc

# 初始化台文轉換器
c = Converter(format='mark')
# 初始化 OpenCC，用於將繁體中文轉為台灣慣用詞
cc = opencc.OpenCC('t2s.json')

# --- 核心修正：在字元集中加入第八聲的聲調符號 "̍" ---
_TAIWANESE_PHONEMES_RE = re.compile(
    r"([a-zA-Zāáàâa̍ēéèêe̍īíìîōóòôo̍ūúùûu̍mngN\d̍]+)", # 在這裡加入了 ̍
    re.IGNORECASE,
)
# --- 修正結束 ---

def an_phah(text: str) -> list[str]:
    """將臺羅拼音語句切分為個別音節 (Phonemes)"""
    text = re.sub(r'\s+', ' ', text).strip()
    prons = _TAIWANESE_PHONEMES_RE.findall(text)
    # --- 核心修正：將所有切分出來的音素轉為小寫再返回 ---
    return [p.lower() for p in prons if p]
    # --- 修正結束 ---

def spell(text: str) -> str:
    """將一句台語漢字轉換為臺羅拼音，並增加異常處理與穩健性"""
    text = cc.convert(text)

    try:
        converted_text = c.get(text)
        if not converted_text:
            return ""
        
        # 使用正規表示式，更穩健地清理連字號和多餘空格
        cleaned_output = re.sub(r'[- ]+', ' ', converted_text).strip()
        return cleaned_output

    except IndexError:
        print(f"警告：Taibun 函式庫在處理文本 '{text}' 時發生內部錯誤，已跳過。")
        return ""
    except Exception as e:
        print(f"警告：Taibun 函式庫在處理文本 '{text}' 時發生未知錯誤: {e}，已跳過。")
        return ""

def taiwanese_to_phonemes(text: str) -> tuple[list[str], list[int], str]:
    """將台語漢字文本轉換為音素序列，並計算 word2ph"""
    norm_text = spell(text)
    phones = an_phah(norm_text)
    # 這個 word2ph 的計算可能需要根據您的模型做調整，但目前暫時維持原樣
    word2ph = [1] * len(phones)
    return phones, word2ph, norm_text