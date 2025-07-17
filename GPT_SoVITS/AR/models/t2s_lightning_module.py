# /GPT_SoVITS/AR/models/t2s_lightning_module.py (終極完整修正版)

import os
import sys
from typing import Dict

# 確保模組路徑正確
now_dir = os.getcwd()
sys.path.append(now_dir)

import torch
from pytorch_lightning import LightningModule
from AR.models.t2s_model import Text2SemanticDecoder
from AR.modules.lr_schedulers import WarmupCosineLRSchedule
from AR.modules.optim import ScaledAdam

class Text2SemanticLightningModule(LightningModule):
    def __init__(self, config, output_dir, is_train=True):
        super().__init__()
        self.config = config
        self.top_k = 3
        # 根據 s1_train.py 傳入的、已修正的 config 初始化模型
        self.model = Text2SemanticDecoder(config=config["model"], top_k=self.top_k) 

        if is_train:
            self.automatic_optimization = False
            self.save_hyperparameters()
            if output_dir:
                self.eval_dir = output_dir / "eval"
                self.eval_dir.mkdir(parents=True, exist_ok=True)

            # 【關鍵修正】使用最穩健的「先刪除再載入」策略
            pretrained_s1 = self.config["train"].get("pretrained_s1")
            if pretrained_s1 and os.path.exists(pretrained_s1):
                print(f"正在載入官方預訓練 GPT 模型: {pretrained_s1}")
                try:
                    ckpt = torch.load(pretrained_s1, map_location="cpu")
                    pretrain_dict = ckpt.get("state_dict", ckpt.get("weight", {}))

                    # 取得當前模型的狀態字典，以便知道哪些鍵是合法的
                    model_dict = self.state_dict()

                    # 刪除預訓練權重中，所有與當前模型結構不匹配的鍵
                    # 特別是尺寸不符的 ar_text_embedding.word_embeddings.weight
                    final_pretrain_dict = {}
                    for k, v in pretrain_dict.items():
                        # 去掉 lightning module 可能自動添加的 'model.' 前綴
                        model_k = k[6:] if k.startswith("model.") else k
                        if model_k in model_dict and model_dict[model_k].shape == v.shape:
                            final_pretrain_dict[k] = v
                        else:
                            print(f"捨棄不匹配或不存在的權重: {k}")

                    # 使用 strict=False 來載入這個經過清理的權重字典
                    self.load_state_dict(final_pretrain_dict, strict=False)
                    print("✅ 已成功跳過不匹配的權重，並載入預訓練 GPT 模型。")

                except Exception as e:
                    print(f"❌ 載入預訓練 GPT 模型失敗，將從頭開始訓練。錯誤: {e}")

    def training_step(self, batch: Dict, batch_idx: int):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        forward = self.model.forward if self.config["train"].get("if_dpo", False) else self.model.forward_old
        loss, acc = forward(
            batch["phoneme_ids"], batch["phoneme_ids_len"],
            batch["semantic_ids"], batch["semantic_ids_len"],
            batch["bert_feature"]
        )
        self.manual_backward(loss)
        if batch_idx > 0 and batch_idx % 4 == 0:
            opt.step()
            opt.zero_grad()
            scheduler.step()

        self.log("total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"top_{self.top_k}_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        model_parameters = self.model.parameters()
        parameters_names = [[name_param_pair[0] for name_param_pair in self.model.named_parameters()]]
        lm_opt = ScaledAdam(
            model_parameters, lr=0.01, betas=(0.9, 0.95), clipping_scale=2.0,
            parameters_names=parameters_names, show_dominant_parameters=False, clipping_update_period=1000
        )
        return {
            "optimizer": lm_opt,
            "lr_scheduler": {
                "scheduler": WarmupCosineLRSchedule(
                    lm_opt, init_lr=self.config["optimizer"]["lr_init"], peak_lr=self.config["optimizer"]["lr"],
                    end_lr=self.config["optimizer"]["lr_end"], warmup_steps=self.config["optimizer"]["warmup_steps"],
                    total_steps=self.config["optimizer"]["decay_steps"]
                )
            },
        }