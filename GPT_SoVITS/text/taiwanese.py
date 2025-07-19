# text/taiwanese.py
import re
from typing import List, Tuple, Optional, Set

# 導入專案既有的英文 G2P 模組
from text import english
from text.taiwanese_symbols import TaiwaneseHokkienSymbols

class TaiwaneseG2P:
    """
    一個能處理台英混合文本的 G2P 處理器。
    """
    def __init__(self) -> None:
        self.symbols_obj = TaiwaneseHokkienSymbols()
        self.symbols: Set[str] = set(self.symbols_obj.symbols)
        self.tone_re = re.compile(r"([0-9])$")

    def _split_syllable(self, syllable: str) -> List[str]:
        """(內部函式) 將單個台羅拼音音節拆分為音素。"""
        phones: List[str] = []
        tone_match = self.tone_re.search(syllable)
        
        if tone_match:
            tone = tone_match.group(1)
            syllable_without_tone = syllable[: -len(tone)]
        else:
            tone = "0"
            syllable_without_tone = syllable
        
        remaining = syllable_without_tone
        while remaining:
            matched = False
            for length in range(min(len(remaining), 3), 0, -1):
                candidate = remaining[:length]
                if candidate in self.symbols:
                    phones.append(candidate)
                    remaining = remaining[length:]
                    matched = True
                    break
            if not matched:
                print(f"[Warning] Cannot find phoneme for '{remaining[0]}' in Taigi syllable '{syllable}'. Skipping.")
                remaining = remaining[1:]

        if tone in self.symbols:
            phones.append(tone)
        
        return phones

    def g2p(self, text: str) -> List[str]:
        """將混合語言的句子轉換為音素序列。"""
        all_phones: List[str] = []
        words = re.split(r'(\s+|[.,?!;])', text)

        for word in filter(None, words):
            if word.isspace():
                if ' ' in self.symbols:
                    all_phones.append(' ')
                continue
            
            if word in [',', '.', '?', '!', ';']:
                if word in self.symbols:
                    all_phones.append(word)
                continue

            if re.search(r"[0-9\-]", word):
                syllables = word.split('-')
                for i, syllable in enumerate(syllables):
                    if syllable:
                        all_phones.extend(self._split_syllable(syllable))
            else:
                try:
                    eng_phones = english.g2p(word)
                    all_phones.extend(eng_phones)
                    # This print statement is good for debugging, can be commented out later.
                    # print(f"--- English word '{word}' converted to {eng_phones} ---")
                except Exception as e:
                    print(f"[Error] English G2P failed for '{word}': {e}. Skipping.")
        
        return all_phones


def text_normalize(text: str) -> str:
    """
    標準化文本。
    """
    text = text.lower()
    
    # [核心更新] 再次擴充正規表達式，加入冒號(:)和單智慧引號(‘ ’)的清洗規則
    text = re.sub(r'["\'()\[\]“”‘’:。]', '', text) # 加入全形句號
    
    return text

# --- 公開 API：提供給 cleaner.py 的主要入口點 ---
_g2p_processor = TaiwaneseG2P()

def taiwanese_to_phonemes(text: str) -> Tuple[List[str], Optional[List[int]], str]:
    """提供給 cleaner.py 的主要入口點。"""
    norm_text = text_normalize(text)
    phones = _g2p_processor.g2p(norm_text)
    word2ph = None 
    return phones, word2ph, norm_text