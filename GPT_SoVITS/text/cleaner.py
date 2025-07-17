# cleaner.py
import os
from text import cleaned_text_to_sequence

# 根據環境變數或預設值，動態導入對應版本的 symbols 和模組
# 這裡假設您已將 taiwanese_symbols.py 放入 text/ 目錄
if os.environ.get("version", "v2") == "v1":
    # from text import chinese
    from text.symbols import symbols
else:
    # from text import chinese2 as chinese
    from text.symbols2 import symbols

# 導入台語處理模組
from text import taiwanese

# 特殊符號處理規則 (此處保持不變)
special = [
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
]


def clean_text(text: str, language: str, version: str = None) -> tuple:
    """
    中央文本清理與 G2P 轉換器。
    """
    if version is None:
        version = os.environ.get("version", "v2")
    
    # 根據版本選擇對應的 symbols 和語言模組映射
    if version == "v1":
        current_symbols = __import__("text.symbols", fromlist=["symbols"]).symbols
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english", "tw": "taiwanese"}
    else:
        current_symbols = __import__("text.symbols2", fromlist=["symbols"]).symbols
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean", "yue": "cantonese", "tw": "taiwanese"}

    if language not in language_module_map:
        # 對於不支援的語言，返回空結果或預設值
        print(f"[Warning] Unsupported language: '{language}'. Defaulting to English empty space.")
        language = "en"
        text = " "

    # 處理特殊靜音符號
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol, version)

    # 動態導入對應的語言處理模組
    lang_module_name = language_module_map[language]
    language_module = __import__(f"text.{lang_module_name}", fromlist=[lang_module_name])

    # 文本正規化
    if hasattr(language_module, "text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text = text

    # G2P 轉換
    if language in ["zh", "yue"]:
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    
    # ******** 整合台語處理的新分支 ********
    elif language == "tw":
        # 調用 taiwanese.py 中我們新建立的標準 API
        phones, word2ph, norm_text = taiwanese.taiwanese_to_phonemes(norm_text)
    # *****************************************

    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [","] + phones
        word2ph = None
    else: # 其他語言如 ja, ko
        phones = language_module.g2p(norm_text)
        word2ph = None

    # 檢查音素是否在 symbols 列表中，若無則替換為 UNK
    new_phones = []
    for ph in phones:
        if ph not in current_symbols:
            print(f"### 診斷資訊：發現未知音素 '{ph}'，它不存在於 symbols 中。已將其替換為 'UNK'。 ###")
            new_phones.append("UNK")
        else:
            new_phones.append(ph)
    phones = new_phones
    
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol, version=None):
    # 此函式邏輯保持不變，但應確保 'tw' 在 language_module_map 中
    if version is None:
        version = os.environ.get("version", "v2")
    
    if version == "v1":
        symbols = __import__("text.symbols", fromlist=["symbols"]).symbols
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english", "tw": "taiwanese"}
    else:
        symbols = __import__("text.symbols2", fromlist=["symbols"]).symbols
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean", "yue": "cantonese", "tw": "taiwanese"}
    
    text = text.replace(special_s, ",")
    lang_module_name = language_module_map[language]
    language_module = __import__(f"text.{lang_module_name}", fromlist=[lang_module_name])
    
    # 假設特殊處理不適用於台語，若需要，得另外設計邏輯
    # 此處沿用舊邏輯
    norm_text = language_module.text_normalize(text)
    if language in ['zh', 'yue']:
        phones, word2ph = language_module.g2p(norm_text)
        new_ph = []
        for ph in phones:
            if ph == ",":
                new_ph.append(target_symbol)
            else:
                new_ph.append(ph)
        return new_ph, word2ph, norm_text
    else:
        # 對於台語、英語等，clean_special 的邏輯可能需要重新定義
        # 此處僅作簡單返回
        phones, word2ph, norm_text = clean_text(text, language, version)
        return phones, word2ph, norm_text


def text_to_sequence(text, language, version=None):
    """將文本轉換為數字序列的主接口。"""
    if version is None:
        version = os.environ.get("version", "v2")
    
    phones, _, _ = clean_text(text, language, version)
    return cleaned_text_to_sequence(phones, version)

# --- 測試區塊 ---
if __name__ == "__main__":
    print("--- 測試中文 ---")
    # 假設 chinese2.g2p 存在
    # print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
    
    print("\n--- 測試台語 ---")
    tw_text = "tsa1-boo2, lí-hó--bô?"
    tw_phones, tw_word2ph, tw_norm_text = clean_text(tw_text, "tw")
    print(f"Input: {tw_text}")
    print(f"Phones: {tw_phones}")
    print(f"Word2Ph: {tw_word2ph}")
    print(f"Normalized: {tw_norm_text}")