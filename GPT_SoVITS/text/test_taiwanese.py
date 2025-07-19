# test_taiwanese.py

# 從您提供的 taiwanese.py 中導入主函數
from taiwanese import taiwanese_to_phonemes

# --- 實驗組：測試有問題的字元 ---
text_problem = "有"
print(f"--- 開始測試問題字元：'{text_problem}' ---")
try:
    phones, word2ph, norm_text = taiwanese_to_phonemes(text_problem)
    
    print(f"輸入漢字 (Input Hanzi): '{text_problem}'")
    print(f"Taibun 轉換後的臺羅拼音 (Tâi-lô): '{norm_text}'")
    print(f"最終切分的音素序列 (Phoneme Sequence): {phones}")
    print(f"Word2Ph: {word2ph}\n")

except Exception as e:
    print(f"處理 '{text_problem}' 時發生錯誤: {e}\n")


# --- 對照組：測試一個正常的詞 ---
text_normal = "價值"
print(f"--- 開始測試對照組字元：'{text_normal}' ---")
try:
    phones, word2ph, norm_text = taiwanese_to_phonemes(text_normal)

    print(f"輸入漢字 (Input Hanzi): '{text_normal}'")
    print(f"Taibun 轉換後的臺羅拼音 (Tâi-lô): '{norm_text}'")
    print(f"最終切分的音素序列 (Phoneme Sequence): {phones}")
    print(f"Word2Ph: {word2ph}\n")

except Exception as e:
    print(f"處理 '{text_normal}' 時發生錯誤: {e}\n")