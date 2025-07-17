# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/data/dataset.py
# reference: https://github.com/lifeiteng/vall-e

# sys.path.append("/data/docker/liujing04/gpt-vits/mq-vits-s1bert_no_bert")
import os
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# --- 核心修正：將 text/__init__.py 的核心邏輯直接複製到此處 ---
# 這樣可以保證我們使用的是硬碟上最新的 symbols.py，徹底繞開快取問題
try:
    from text.symbols import symbols
    _symbol_to_id = {s: i for i, s in enumerate(symbols)}
    def cleaned_text_to_sequence(cleaned_text):
        return [_symbol_to_id[symbol] for symbol in cleaned_text]
    print("✅ dataset.py 已成功建立本地字典。")
except Exception as e:
    print(f"❌ dataset.py 無法建立字典，請檢查 text/symbols.py 檔案。錯誤: {e}")
# --- 修正結束 ---

version = os.environ.get("version", None)

# from config import exp_dir


def batch_sequences(sequences: List[np.array], axis: int = 0, pad_value: int = 0):
    seq = sequences[0]
    ndim = seq.ndim
    if axis < 0:
        axis += ndim
    dtype = seq.dtype
    pad_value = dtype.type(pad_value)
    seq_lengths = [seq.shape[axis] for seq in sequences]
    max_length = np.max(seq_lengths)

    padded_sequences = []
    for seq, length in zip(sequences, seq_lengths):
        padding = [(0, 0)] * axis + [(0, max_length - length)] + [(0, 0)] * (ndim - axis - 1)
        padded_seq = np.pad(seq, padding, mode="constant", constant_values=pad_value)
        padded_sequences.append(padded_seq)
    batch = np.stack(padded_sequences)
    return batch


class Text2SemanticDataset(Dataset):
    def __init__(self, phoneme_path: str, semantic_path: str, max_sample: int = None, max_sec: int = 100, pad_val: int = 1024, min_ps_ratio: int = 0.1, max_ps_ratio: int = 1000):
        super().__init__()
        self.semantic_data = pd.read_csv(semantic_path, delimiter="\t", encoding="utf-8")
        self.path2 = phoneme_path
        self.path3 = f"{os.path.dirname(phoneme_path)}/3-bert"
        self.phoneme_data = {}
        with open(self.path2, "r", encoding="utf8") as f:
            for line in f.read().strip("\n").split("\n"):
                try:
                    name, spk, lang, text = line.split("\t")
                    self.phoneme_data[name] = [spk, lang, text]
                except:
                    pass
        self.PAD, self.hz, self.max_sec = pad_val, int(os.environ.get("hz", "25hz")[:-2]), max_sec
        self.min_ps_ratio, self.max_ps_ratio = min_ps_ratio, max_ps_ratio
        if max_sample: self.semantic_data = self.semantic_data[:max_sample]
        self.semantic_phoneme, self.item_names, self.sample_languages = [], [], []
        self.init_batch()
        del self.semantic_data, self.phoneme_data

    def init_batch(self):
        semantic_data_len = len(self.semantic_data)
        phoneme_data_len = len(self.phoneme_data.keys())
        print("semantic_data_len:", semantic_data_len)
        print("phoneme_data_len:", phoneme_data_len)
        print(self.semantic_data)
        idx = 0
        num_not_in = 0
        num_deleted_bigger = 0
        num_deleted_ps = 0
        for i in range(len(self.semantic_data)):
            item_name = self.semantic_data.iloc[i, 0]
            if item_name not in self.phoneme_data: continue
            phoneme, language, text = self.phoneme_data[item_name] # 修正順序
            semantic_ids = [int(idx) for idx in self.semantic_data.iloc[i, 1].split(" ")]
            if len(semantic_ids) > self.max_sec * self.hz: continue
            phoneme_list = phoneme.split(" ")
            try:
                phoneme_ids = cleaned_text_to_sequence(phoneme_list)
            except: continue
            if len(phoneme_ids) > self.max_sec * self.hz / 2.5: continue
            ps_ratio = len(phoneme_ids) / (len(semantic_ids) / self.hz)
            if ps_ratio > self.max_ps_ratio or ps_ratio < self.min_ps_ratio: continue
            self.semantic_phoneme.append((semantic_ids, phoneme_ids))
            self.item_names.append(item_name)
            self.sample_languages.append(language)
        print("dataset.__len__():", len(self))

    def __get_item_names__(self) -> List[str]:
        return self.item_names

    def __len__(self) -> int:
        return len(self.semantic_phoneme)

    def __getitem__(self, idx: int) -> Dict:
        semantic_ids, phoneme_ids = self.semantic_phoneme[idx]
        item_name, language = self.item_names[idx], self.sample_languages[idx]
        bert_feature = None
        if language == "zh":
            path_bert = f"{self.path3}/{item_name}.pt"
            if os.path.exists(path_bert):
                bert_feature = torch.load(path_bert, map_location="cpu")
                if bert_feature.shape[-1] != len(phoneme_ids):
                    bert_feature = None
        return {
            "idx": idx, "phoneme_ids": phoneme_ids, "phoneme_ids_len": len(phoneme_ids),
            "semantic_ids": semantic_ids, "semantic_ids_len": len(semantic_ids),
            "bert_feature": bert_feature,
        }

    def get_sample_length(self, idx: int):
        semantic_ids = self.semantic_phoneme[idx][0]
        sec = 1.0 * len(semantic_ids) / self.hz
        return sec

    def collate(self, examples: List[Dict]) -> Dict:
        sample_index: List[int] = []
        phoneme_ids: List[torch.Tensor] = []
        phoneme_ids_lens: List[int] = []
        semantic_ids: List[torch.Tensor] = []
        semantic_ids_lens: List[int] = []
        # return

        for item in examples:
            sample_index.append(item["idx"])
            phoneme_ids.append(np.array(item["phoneme_ids"], dtype=np.int64))
            semantic_ids.append(np.array(item["semantic_ids"], dtype=np.int64))
            phoneme_ids_lens.append(item["phoneme_ids_len"])
            semantic_ids_lens.append(item["semantic_ids_len"])

        # pad 0
        phoneme_ids = batch_sequences(phoneme_ids)
        semantic_ids = batch_sequences(semantic_ids, pad_value=self.PAD)

        # # convert each batch to torch.tensor
        phoneme_ids = torch.tensor(phoneme_ids)
        semantic_ids = torch.tensor(semantic_ids)
        phoneme_ids_lens = torch.tensor(phoneme_ids_lens)
        semantic_ids_lens = torch.tensor(semantic_ids_lens)
        bert_padded = torch.FloatTensor(len(examples), 1024, max(phoneme_ids_lens))
        bert_padded.zero_()

        for idx, item in enumerate(examples):
            bert = item["bert_feature"]
            if bert != None:
                bert_padded[idx, :, : bert.shape[-1]] = bert

        return {
            # List[int]
            "ids": sample_index,
            # torch.Tensor (B, max_phoneme_length)
            "phoneme_ids": phoneme_ids,
            # torch.Tensor (B)
            "phoneme_ids_len": phoneme_ids_lens,
            # torch.Tensor (B, max_semantic_ids_length)
            "semantic_ids": semantic_ids,
            # torch.Tensor (B)
            "semantic_ids_len": semantic_ids_lens,
            # torch.Tensor (B, 1024, max_phoneme_length)
            "bert_feature": bert_padded,
        }


if __name__ == "__main__":
    root_dir = "/data/docker/liujing04/gpt-vits/prepare/dump_mix/"
    dataset = Text2SemanticDataset(
        phoneme_path=root_dir + "phoneme_train.npy",
        semantic_path=root_dir + "semantic_train.tsv",
    )

    batch_size = 12
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate,
        shuffle=False,
    )
    for i, batch in enumerate(dataloader):
        if i % 1000 == 0:
            print(i)

