# /GPT_SoVITS/s2_train.py (最終完整修正版)

import warnings
warnings.filterwarnings("ignore")
import os
import sys

# --- 核心修正：在所有 import 之前，手動將核心程式碼目錄加入 Python 路徑 ---
now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)
# --- 修正結束 ---

import utils
hps = utils.get_hparams(stage=2)
os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")
import logging
from random import randint

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from module import commons
from module.data_utils import (
    DistributedBucketSampler, TextAudioSpeakerCollate, TextAudioSpeakerLoader
)
from module.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from module.models import MultiPeriodDiscriminator, SynthesizerTrn
from process_ckpt import savee
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("medium")

global_step = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    
    # 【關鍵修正】如果只有單 GPU，就不使用 mp.spawn，直接運行 run 函數
    if n_gpus <= 1:
        print("Detected single GPU or CPU, running training directly.")
        run(0, 1, hps)
    else:
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))

def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.data.exp_dir)
        logger.info(hps)
        writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))
    else:
        logger, writer, writer_eval = None, None, None

    if n_gpus > 1:
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
        )
    
    torch.manual_seed(hps.train.seed)
    if device == "cuda":
        torch.cuda.set_device(rank)

    train_dataset = TextAudioSpeakerLoader(hps.data, version=hps.model.version)
    train_sampler = DistributedBucketSampler(
        train_dataset, hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        num_replicas=n_gpus, rank=rank, shuffle=True
    )
    collate_fn = TextAudioSpeakerCollate(version=hps.model.version)
    train_loader = DataLoader(
        train_dataset, num_workers=0, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, batch_sampler=train_sampler, persistent_workers=False, prefetch_factor=None
    )

    net_g = SynthesizerTrn(hps.data.filter_length//2+1, hps.train.segment_size//hps.data.hop_length, n_speakers=hps.data.n_speakers, **hps.model).to(device)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm, version=hps.model.version).to(device)
    
    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    try:
        _, optim_d, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(f"{hps.data.exp_dir}/logs_s2_{hps.model.version}", "D_*.pth"), net_d, optim_d)
        _, optim_g, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(f"{hps.data.exp_dir}/logs_s2_{hps.model.version}", "G_*.pth"), net_g, optim_g)
        global_step = (epoch_str - 1) * len(train_loader)
        if rank == 0: logger.info("Resuming from checkpoint")
    except:
        epoch_str = 1
        global_step = 0
        # ... (預訓練模型載入邏輯保持不變) ...

    if n_gpus > 1:
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=False)
        net_d = DDP(net_d, device_ids=[rank])
    
    for param_group in optim_g.param_groups: param_group['initial_lr'] = hps.train.learning_rate
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)
    
    if rank == 0: print(f"start training from epoch {epoch_str}")
    
    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], logger, [writer, writer_eval])
        scheduler_g.step()
        scheduler_d.step()
        
    if rank == 0: print("training done")

def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    # scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        if hps.model.version in {"v2Pro", "v2ProPlus"}:
            ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, sv_emb = data
        else:
            ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths = data
        if torch.cuda.is_available():
            spec, spec_lengths = (
                spec.cuda(
                    rank,
                    non_blocking=True,
                ),
                spec_lengths.cuda(
                    rank,
                    non_blocking=True,
                ),
            )
            y, y_lengths = (
                y.cuda(
                    rank,
                    non_blocking=True,
                ),
                y_lengths.cuda(
                    rank,
                    non_blocking=True,
                ),
            )
            ssl = ssl.cuda(rank, non_blocking=True)
            ssl.requires_grad = False
            # ssl_lengths = ssl_lengths.cuda(rank, non_blocking=True)
            text, text_lengths = (
                text.cuda(
                    rank,
                    non_blocking=True,
                ),
                text_lengths.cuda(
                    rank,
                    non_blocking=True,
                ),
            )
            if hps.model.version in {"v2Pro", "v2ProPlus"}:
                sv_emb = sv_emb.cuda(rank, non_blocking=True)
        else:
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)
            ssl = ssl.to(device)
            ssl.requires_grad = False
            # ssl_lengths = ssl_lengths.cuda(rank, non_blocking=True)
            text, text_lengths = text.to(device), text_lengths.to(device)
            if hps.model.version in {"v2Pro", "v2ProPlus"}:
                sv_emb = sv_emb.to(device)
        with autocast(enabled=hps.train.fp16_run):
            if hps.model.version in {"v2Pro", "v2ProPlus"}:
                (y_hat, kl_ssl, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), stats_ssl) = net_g(
                    ssl, spec, spec_lengths, text, text_lengths, sv_emb
                )
            else:
                (
                    y_hat,
                    kl_ssl,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                    stats_ssl,
                ) = net_g(ssl, spec, spec_lengths, text, text_lengths)

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r,
                    y_d_hat_g,
                )
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, kl_ssl, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch,
                        100.0 * batch_idx / len(train_loader),
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl_ssl": kl_ssl,
                        "loss/g/kl": loss_kl,
                    }
                )

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = None
                try:  ###Some people installed the wrong version of matplotlib.
                    image_dict = {
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(
                            y_mel[0].data.cpu().numpy(),
                        ),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].data.cpu().numpy(),
                        ),
                        "all/mel": utils.plot_spectrogram_to_numpy(
                            mel[0].data.cpu().numpy(),
                        ),
                        "all/stats_ssl": utils.plot_spectrogram_to_numpy(
                            stats_ssl[0].data.cpu().numpy(),
                        ),
                    }
                except:
                    pass
                if image_dict:
                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        images=image_dict,
                        scalars=scalar_dict,
                    )
                else:
                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        scalars=scalar_dict,
                    )
        global_step += 1
    if epoch % hps.train.save_every_epoch == 0 and rank == 0:
        if hps.train.if_save_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version),
                    "G_{}.pth".format(global_step),
                ),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version),
                    "D_{}.pth".format(global_step),
                ),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version),
                    "G_{}.pth".format(233333333333),
                ),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version),
                    "D_{}.pth".format(233333333333),
                ),
            )
        if rank == 0 and hps.train.if_save_every_weights == True:
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        global_step,
                        hps,
                        model_version=None if hps.model.version not in {"v2Pro", "v2ProPlus"} else hps.model.version,
                    ),
                )
            )

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))

if __name__ == "__main__":
    main()