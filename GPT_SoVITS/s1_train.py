# # /GPT_SoVITS/s1_train.py (最終完整修正版)
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
gpt_sovits_core_path = os.path.join(now_dir, "GPT_SoVITS")
if gpt_sovits_core_path not in sys.path:
    sys.path.insert(0, gpt_sovits_core_path)

import argparse
import logging
import platform
from pathlib import Path
from collections import OrderedDict

import torch
from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from AR.utils import get_newest_ckpt
from process_ckpt import my_save
from text.symbols import symbols

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")

class my_model_ckpt(ModelCheckpoint):
    def __init__(self, config, if_save_latest, if_save_every_weights, half_weights_save_dir, exp_name, **kwargs):
        super().__init__(**kwargs)
        self.if_save_latest, self.if_save_every_weights = if_save_latest, if_save_every_weights
        self.half_weights_save_dir, self.exp_name, self.config = half_weights_save_dir, exp_name, config
    def on_train_epoch_end(self, trainer, pl_module):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if self.if_save_latest: to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest:
                    for name in to_clean:
                        try: os.remove(f"{self.dirpath}/{name}")
                        except: pass
                if self.if_save_every_weights:
                    to_save_od = OrderedDict([("weight", OrderedDict()), ("config", self.config), ("info", f"GPT-e{trainer.current_epoch + 1}")])
                    dictt = trainer.strategy._lightning_module.state_dict()
                    for key in dictt: to_save_od["weight"][key] = dictt[key].half()
                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        my_save(to_save_od, f"{self.half_weights_save_dir}/{self.exp_name}-e{trainer.current_epoch + 1}.ckpt")
            self._save_last_checkpoint(trainer, monitor_candidates)

def main(args):
    config = load_yaml_config(args.config_file)
    n_phoneme = len(symbols)
    config["model"]["n_phoneme"] = n_phoneme
    print(f"✅ 已動態將 GPT 模型的字典大小 (n_phoneme) 設定為: {n_phoneme}")
    
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config["train"]["seed"], workers=True)
    
    ckpt_callback = my_model_ckpt(
        config=config, if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"], save_top_k=-1, save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"], dirpath=ckpt_dir
    )
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    os.environ["MASTER_ADDR"], os.environ["USE_LIBUV"] = "localhost", "0"
    
    trainer = Trainer(
        max_epochs=config["train"]["epochs"], accelerator="cuda" if torch.cuda.is_available() else "cpu",
        limit_val_batches=0, devices="auto", benchmark=False, fast_dev_run=False,
        strategy=DDPStrategy(process_group_backend="nccl" if platform.system() != "Windows" else "gloo") if torch.cuda.device_count() > 1 else "auto",
        precision=config["train"]["precision"], logger=logger, num_sanity_val_steps=0,
        callbacks=[ckpt_callback], use_distributed_sampler=False
    )

    model = Text2SemanticLightningModule(config, output_dir, is_train=True)
    data_module = Text2SemanticDataModule(
        config, train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"]
    )

    try:
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="GPT_SoVITS/configs/s1longer-v2.yaml", help="path of config file")
    args = parser.parse_args()
    logging.info(str(args))
    main(args)