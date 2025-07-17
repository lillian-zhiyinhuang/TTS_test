# /GPT_SoVITS/AR/data/data_module.py (終極完整修正版)

from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from AR.data.dataset import Text2SemanticDataset

class Text2SemanticDataModule(LightningDataModule):
    def __init__(
        self,
        config,
        train_semantic_path,
        train_phoneme_path,
        dev_semantic_path=None,
        dev_phoneme_path=None,
        is_train=True
    ):
        super().__init__()
        self.config = config
        self.train_semantic_path = train_semantic_path
        self.train_phoneme_path = train_phoneme_path
        self.dev_semantic_path = dev_semantic_path
        self.dev_phoneme_path = dev_phoneme_path
        self.is_train = is_train

    def setup(self, stage: Optional[str] = None):
      if self.is_train:
          # --- 開始終極修正 ---
          # 從 self.config 中，精確地取出 Text2SemanticDataset 需要的每一個參數
          self._train_dataset = Text2SemanticDataset(
              phoneme_path=self.train_phoneme_path,
              semantic_path=self.train_semantic_path,
              max_sec=self.config["data"].get("max_sec", 100),
              pad_val=self.config["data"].get("pad_val", 1024),
              # min_ps_ratio 和 max_ps_ratio 我們之前已在 dataset.py 中設定了寬鬆的預設值，此處無需傳遞
          )
          # --- 修正結束 ---
        # if self.dev_semantic_path is not None:
        #     self._dev_dataset = Text2SemanticDataset(
        #         config=self.config,
        #         semantic_path=self.dev_semantic_path,
        #         phoneme_path=self.dev_phoneme_path,
        #         is_val=True
        #     )

    def train_dataloader(self):
        # 【關鍵修正】在這裡加入與 s2_train.py 同樣的參數修正
        return DataLoader(
            self._train_dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            sampler=None,
            collate_fn=self._train_dataset.collate,
            # 確保以下參數與 num_workers=0 兼容
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=None,
        )

    def val_dataloader(self):
        # if self.dev_semantic_path is not None:
        #     return DataLoader(
        #         self._dev_dataset,
        #         batch_size=1,
        #         shuffle=False,
        #         collate_fn=self._dev_dataset.collate,
        #         num_workers=0,
        #         persistent_workers=False,
        #         prefetch_factor=None
        #     )
        # else:
        return None # 或者返回一個空的 DataLoader