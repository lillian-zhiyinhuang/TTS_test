# /GPT_SoVITS/AR/models/t2s_model.py (最終完整修正版)

import math
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from AR.models.utils import (
    dpo_loss, get_batch_logps, make_pad_mask, make_pad_mask_left,
    make_reject_y, sample, topk_sampling
)
from AR.modules.embedding import (SinePositionalEmbedding,
                                  TokenEmbedding)
from AR.modules.transformer import (LayerNorm, TransformerEncoder,
                                    TransformerEncoderLayer)

# JIT 相關的輔助類別和函數...
@torch.jit.script
class T2SMLP:
    def __init__(self, w1, b1, w2, b2):
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2
    def forward(self, x):
        x = F.relu(F.linear(x, self.w1, self.b1))
        return F.linear(x, self.w2, self.b2)

@torch.jit.script
class T2SBlock:
    def __init__(self, num_heads, hidden_dim: int, mlp: T2SMLP, qkv_w, qkv_b, out_w, out_b, norm_w1, norm_b1, norm_eps1, norm_w2, norm_b2, norm_eps2):
        self.num_heads, self.hidden_dim, self.mlp = num_heads, hidden_dim, mlp
        self.qkv_w, self.qkv_b, self.out_w, self.out_b = qkv_w, qkv_b, out_w, out_b
        self.norm_w1, self.norm_b1, self.norm_eps1 = norm_w1, norm_b1, norm_eps1
        self.norm_w2, self.norm_b2, self.norm_eps2 = norm_w2, norm_b2, norm_eps2

    def process_prompt(self, x: torch.Tensor, attn_mask: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)
        bsz, q_len, kv_len = q.shape[0], q.shape[1], k.shape[1]
        k_cache, v_cache = k, v
        q, k, v = (t.view(bsz, t.shape[1], self.num_heads, -1).transpose(1, 2) for t in (q, k_cache, v_cache))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=~attn_mask)
        attn = attn.transpose(1, 2).reshape(bsz, q_len, -1)
        attn = F.linear(attn, self.out_w, self.out_b)
        x = x + attn
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1)
        x = x + self.mlp.forward(x)
        return F.layer_norm(x, [self.hidden_dim], self.norm_w2, self.norm_b2, self.norm_eps2), k_cache, v_cache

    def decode_next_token(self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)
        k_cache, v_cache = torch.cat([k_cache, k], dim=1), torch.cat([v_cache, v], dim=1)
        bsz, q_len, kv_len = q.shape[0], q.shape[1], k_cache.shape[1]
        q, k, v = (t.view(bsz, t.shape[1], self.num_heads, -1).transpose(1, 2) for t in (q, k_cache, v_cache))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(bsz, q_len, -1)
        attn = F.linear(attn, self.out_w, self.out_b)
        x = x + attn
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1)
        x = x + self.mlp.forward(x)
        return F.layer_norm(x, [self.hidden_dim], self.norm_w2, self.norm_b2, self.norm_eps2), k_cache, v_cache
        
@torch.jit.script
class T2STransformer:
    def __init__(self, num_blocks: int, blocks: List[T2SBlock]):
        self.num_blocks, self.blocks = num_blocks, blocks
    def process_prompt(self, x: torch.Tensor, attn_mask: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        k_cache, v_cache = [], []
        for block in self.blocks:
            x, k, v = block.process_prompt(x, attn_mask, padding_mask)
            k_cache.append(k)
            v_cache.append(v)
        return x, k_cache, v_cache
    def decode_next_token(self, x: torch.Tensor, k_cache: List[torch.Tensor], v_cache: List[torch.Tensor], attn_mask: Optional[torch.Tensor] = None):
        for i, block in enumerate(self.blocks):
            x, k_cache[i], v_cache[i] = block.decode_next_token(x, k_cache[i], v_cache[i], attn_mask)
        return x, k_cache, v_cache

class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.top_k = top_k
        self.model_dim = config.get("hidden_dim", 512)
        self.embedding_dim = config.get("embedding_dim", self.model_dim)
        self.num_head = config.get("n_head", config.get("head", 8))
        self.num_layers = config.get("n_layer", config.get("num_layers", 12))
        self.f_ffn_dim = config.get("ffn_dim", self.model_dim * 4)
        self.n_phoneme = config.get("n_phoneme", config.get("phoneme_vocab_size"))
        self.phoneme_vocab_size = self.n_phoneme
        self.vocab_size = config.get("vocab_size", 1025)
        self.p_dropout = config.get("dropout", 0.0)
        self.EOS = config.get("EOS", 1024)
        self.norm_first = config.get("norm_first", norm_first)
        assert self.EOS == self.vocab_size - 1
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size, self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim, nhead=self.num_head, dim_feedforward=self.f_ffn_dim,
                dropout=0.1, batch_first=True, norm_first=self.norm_first
            ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if self.norm_first else None,
        )
        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")
        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size, top_k=top_k, average="micro",
            multidim_average="global", ignore_index=self.EOS
        )

    def forward_old(self, x, x_lens, y, y_lens, bert_feature):
        x = self.ar_text_embedding(x)
        if bert_feature is not None and bert_feature.shape[-1] == x.shape[-2]:
             x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)
        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len, y_len = x_lens.max(), y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)
        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        x_attn_mask = F.pad(torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device), (0, y_len), value=True)
        y_attn_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=x.device), diagonal=1), (x_len, 0), value=False)
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = xy_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_head, -1, -1).reshape(bsz * self.num_head, 1, src_len)
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_pos = torch.concat([x, y_pos], dim=1)
        xy_dec, _ = self.h((xy_pos, None), mask=new_attn_mask)
        logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(0, 2, 1)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        acc = self.ar_accuracy_metric(logits.detach(), targets).item()
        return loss, acc
    
    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(y_mask_int, (0, 1), value=1)
        return targets[:, :-1], targets[:, 1:]
    
    # --- 【關鍵修正】補回所有缺失的 infer 相關函數 ---
    def infer_panel(self, x, x_lens, prompts, bert_feature, top_k: int = -100, top_p: int = 100, early_stop_num: int = -1, temperature: float = 1.0, **kwargs):
        return self.infer_panel_naive(x, x_lens, prompts, bert_feature, top_k, top_p, early_stop_num, temperature, **kwargs)

    def infer_panel_naive(self, x, x_lens, prompts, bert_feature, top_k: int, top_p: int, early_stop_num: int, temperature: float, **kwargs):
        # x = self.ar_text_embedding(x)
        # if bert_feature is not None:
        #      x = x + self.bert_proj(bert_feature.transpose(1, 2))
        # x = self.ar_text_position(x)

        x = self.ar_text_embedding(x)
        # 【最終修正】在推理時，也加上同樣的、最穩健的檢查
        if bert_feature is not None and bert_feature.shape[-1] == x.shape[-2]:
             x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        y = prompts
        prefix_len = y.shape[1] if y is not None else 0
        x_len = x.shape[1]
        stop = False
        for _ in tqdm(range(1500)):
            if y is None:
                y_emb = self.ar_audio_embedding(torch.zeros(1, 0, dtype=torch.long, device=x.device))
            else:
                y_emb = self.ar_audio_embedding(y)
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            y_len = y.shape[1] if y is not None else 0
            
            x_attn_mask = F.pad(torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device), (0, y_len), value=True)
            y_attn_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=x.device), diagonal=1), (x_len, 0), value=False)
            xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)

            xy_dec, _ = self.h((xy_pos, None), mask=xy_attn_mask)
            logits = self.ar_predict_layer(xy_dec[:, -1])
            
            samples = sample(logits, y, top_k=top_k, top_p=top_p, repetition_penalty=1.0, temperature=temperature)[0]
            
            if y is None:
                y = samples
            else:
                y = torch.concat([y, samples], dim=1)

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                break
        return y, 0