"""
NCA 预训练脚本 - 从预生成的 .npy 文件加载数据
适配 StructGen-rs 生成的批量数据格式
完全独立，不依赖其他模块
"""

import os
import sys
import argparse
import glob
import math
import logging
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_

import numpy as np
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger('nca_train_npy')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ===== Tokenizer (NumPy 版本) ===== #

class NCA_Tokenizer:
    """NCA Tokenizer - NumPy 实现"""
    
    def __init__(self, patch: int, num_colors: int = 10):
        self.patch = patch
        self.num_colors = num_colors
        self.start_tk = num_colors ** (patch ** 2)
        self.end_tk = num_colors ** (patch ** 2) + 1
    
    def encode_task(self, grid: np.ndarray):
        """
        编码网格数据为 token 序列
        
        参数:
            grid: (B, T, H, W, C) 或 (B, T, H, W) 形状的网格数据
                  如果是 (B, T, H, W, C)，则 C 是 one-hot 编码的状态
                  如果是 (B, T, H, W)，则每个值是状态索引
            
        返回:
            seq: (B, seq_len) token 序列
            targets: (B, seq_len) 目标序列
        """
        B, N = grid.shape[0], grid.shape[1]
        
        # 处理 one-hot 格式：将 (B, T, H, W, C) 转换为 (B, T, H, W)
        if grid.ndim == 5:
            # one-hot 格式，取 argmax 得到状态索引
            grid = np.argmax(grid, axis=-1)
        
        H, W = grid.shape[2], grid.shape[3]
        grid = grid.astype(np.int64)
        
        N_H = H // self.patch
        N_W = W // self.patch
        
        # 重塑为 patches
        grid = grid.reshape(B, N, N_H, self.patch, N_W, self.patch)
        grid = grid.transpose(0, 1, 2, 4, 3, 5)
        grid = grid.reshape(B, N, N_H * N_W, self.patch * self.patch)
        
        # 转换为 tokens
        powers = self.num_colors ** np.arange(self.patch * self.patch)
        tokens = np.einsum('bnlp,p->bnl', grid, powers)
        target = tokens.copy()
        
        # 添加 start/end tokens
        mask = np.full((B, N, 1), -100, dtype=tokens.dtype)
        start_tokens = np.full((B, N, 1), self.start_tk, dtype=tokens.dtype)
        end_tokens = np.full((B, N, 1), self.end_tk, dtype=tokens.dtype)
        
        tokens = np.concatenate([start_tokens, tokens, end_tokens], axis=-1)
        target = np.concatenate([mask, target, mask], axis=-1)
        
        tokens = tokens.reshape(B, -1)
        target = target.reshape(B, -1)
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# ===== 数据集定义 ===== #

class NpyNCADataset(Dataset):
    """从 .npy 文件加载的 NCA 数据集"""
    
    def __init__(self, 
                 data_dir: str,
                 pattern: str = "nca2d_batch_*.npy",
                 max_samples: Optional[int] = None,
                 max_seq_len: int = 1024,
                 min_grid: int = 1,
                 patch: int = 3,
                 num_colors: int = 10):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.min_grid = min_grid
        self.patch = patch
        self.num_colors = num_colors
        self.grid_len = (12 // patch) ** 2 + 2
        
        # 加载所有 .npy 文件
        self.files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not self.files:
            raise ValueError(f"在 {data_dir} 中未找到匹配 {pattern} 的文件")
        
        log.info(f"找到 {len(self.files)} 个数据文件")
        
        # 加载并合并数据
        all_data = []
        total_samples = 0
        for f in self.files:
            data = np.load(f)
            all_data.append(data)
            total_samples += data.shape[0]
            if max_samples and total_samples >= max_samples:
                break
        
        self.data = np.concatenate(all_data, axis=0)
        if max_samples:
            self.data = self.data[:max_samples]
        
        log.info(f"加载了 {self.data.shape[0]} 个样本，形状: {self.data.shape}")
        
        # 使用 tokenizer 编码
        self.tokenizer = NCA_Tokenizer(patch, num_colors=num_colors)
        self.seq, self.targets = self._encode_data()
        
    def _encode_data(self):
        seq, targets = self.tokenizer.encode_task(self.data)
        return seq, targets
    
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        seq = self.seq[idx].clone()
        targets = self.targets[idx].clone()
        
        # mask out negative tokens
        target = torch.where(seq < 0, torch.tensor(-100, dtype=seq.dtype), seq)
        target[:self.min_grid * self.grid_len] = -100
        
        # shift
        seq_out = seq[:-1]
        targets_out = target[1:]
        
        # pad/truncate
        if seq_out.shape[0] > self.max_seq_len:
            seq_out = seq_out[:self.max_seq_len]
            targets_out = targets_out[:self.max_seq_len]
        elif seq_out.shape[0] < self.max_seq_len:
            pad_len = self.max_seq_len - seq_out.shape[0]
            seq_out = torch.cat([seq_out, torch.full((pad_len,), -100, dtype=seq_out.dtype)])
            targets_out = torch.cat([targets_out, torch.full((pad_len,), -100, dtype=targets_out.dtype)])
        
        return seq_out.float(), targets_out.long()


# ===== LLaMA 模型定义 ===== #

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn = attn + mask
        
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_head: int, hidden_dim: int):
        super().__init__()
        self.attn = Attention(dim, n_head)
        self.ffn = FeedForward(dim, hidden_dim)
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LLaMA(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_layer: int, n_head: int, seq_len: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_head, dim * 4)
            for _ in range(n_layer)
        ])
        
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        self.seq_len = seq_len
    
    def forward(self, x, mask=None):
        B, T = x.shape
        
        # 处理负值 token
        x = torch.clamp(x, min=0).long()
        
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        h = self.tok_emb(x) + self.pos_emb(pos)
        
        for layer in self.layers:
            h = layer(h, mask)
        
        h = self.norm(h)
        return self.head(h)


class DownstreamModel(nn.Module):
    """下游任务模型，包装 LLaMA"""
    
    def __init__(self, vocab_size: int, dim: int, n_layer: int, n_head: int, 
                 seq_len: int, patch: int, num_colors: int, input_bias: float = 0.1):
        super().__init__()
        self.patch = patch
        self.num_colors = num_colors
        self.input_dim = patch ** 2
        
        # 输入投影层
        self.input_proj = nn.Linear(self.input_dim, dim)
        self.input_bias = nn.Parameter(torch.ones(dim) * input_bias)
        
        # LLaMA 主干
        self.backbone = LLaMA(vocab_size, dim, n_layer, n_head, seq_len)
        
        # 输出投影层
        self.output_proj = nn.Linear(dim, vocab_size)
    
    def forward(self, x, mask=None):
        # x: (B, T) 输入 token 序列
        B, T = x.shape
        
        # 处理负值
        x = torch.clamp(x, min=0).long()
        
        # 通过主干网络
        logits = self.backbone(x, mask)
        
        return logits


def create_attention_mask(seq_len: int, additive: bool = True):
    """创建因果注意力掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    if additive:
        mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)


# ===== 训练函数 ===== #

def train(args):
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 计算 seq_len 和 vocab_size
    grid_len = (args.grid // args.patch) ** 2 + 2
    num_examples = 10
    args.seq_len = num_examples * grid_len
    
    # vocab_size 必须足够大以容纳所有可能的 token
    # 最大 token 值 = num_colors^(patch^2) + 1 (end token)
    args.vocab_size = args.num_colors ** (args.patch ** 2) + 2
    
    log.info(f"序列长度: {args.seq_len}, grid_len: {grid_len}")
    log.info(f"词汇表大小: {args.vocab_size}")
    
    # 加载数据
    log.info(f"数据目录: {args.data_dir}")
    
    train_dataset = NpyNCADataset(
        data_dir=args.data_dir,
        pattern="nca2d_batch_*.npy",
        max_samples=args.train_samples,
        max_seq_len=args.seq_len,
        min_grid=args.min_grid,
        patch=args.patch,
        num_colors=args.num_colors
    )
    
    val_dataset = NpyNCADataset(
        data_dir=args.data_dir,
        pattern="nca2d_batch_*.npy",
        max_samples=args.val_samples,
        max_seq_len=args.seq_len,
        min_grid=args.min_grid,
        patch=args.patch,
        num_colors=args.num_colors
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    log.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 创建模型
    model = DownstreamModel(
        vocab_size=args.vocab_size,
        dim=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        seq_len=args.seq_len,
        patch=args.patch,
        num_colors=args.num_colors,
        input_bias=args.input_bias
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"模型参数量: {total_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 学习率调度器
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0 - step / total_steps
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 损失函数
    criterion = CrossEntropyLoss(ignore_index=-100)
    
    # 训练循环
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            seq, targets = batch
            seq = seq.to(device)
            targets = targets.to(device)
            
            # 创建注意力掩码
            B, T = seq.shape
            attn_mask = create_attention_mask(T).expand(B, -1, -1, -1).to(device)
            
            # 前向传播
            logits = model(seq, attn_mask)
            
            # 计算损失
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            loss = criterion(logits, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            if args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
            
            # 验证
            if global_step % args.val_freq == 0:
                val_loss = validate(model, val_loader, criterion, device, args.seq_len)
                log.info(f"Step {global_step}: val_loss = {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, os.path.join(args.save_dir, 'best_model.pt'))
                    log.info(f"保存最佳模型: val_loss = {val_loss:.4f}")
                
                model.train()
        
        avg_loss = epoch_loss / num_batches
        log.info(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
        
        # 保存 checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    log.info("训练完成!")


def validate(model, val_loader, criterion, device, seq_len):
    """验证函数"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            seq, targets = batch
            seq = seq.to(device)
            targets = targets.to(device)
            
            B, T = seq.shape
            attn_mask = create_attention_mask(T).expand(B, -1, -1, -1).to(device)
            
            logits = model(seq, attn_mask)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            
            loss = criterion(logits, targets)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


# ===== 参数定义 ===== #

@dataclass
class Args:
    seed: int = 42
    device: str = 'cuda:0'
    
    data_dir: str = ''
    train_samples: int = 10000
    val_samples: int = 1000
    
    grid: int = 12
    num_colors: int = 10
    patch: int = 3
    min_grid: int = 1
    
    vocab_size: int = 16384
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    input_bias: float = 0.1
    seq_len: int = 1024
    
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-5
    grad_clip: float = 1.0
    val_freq: int = 100
    
    save_dir: str = 'data/model/nca_npy'


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_samples', type=int, default=10000)
    parser.add_argument('--val_samples', type=int, default=1000)
    
    parser.add_argument('--grid', type=int, default=12)
    parser.add_argument('--num_colors', type=int, default=10)
    parser.add_argument('--patch', type=int, default=3)
    parser.add_argument('--min_grid', type=int, default=1)
    
    parser.add_argument('--vocab_size', type=int, default=16384)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--input_bias', type=float, default=0.1)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--val_freq', type=int, default=100)
    
    parser.add_argument('--save_dir', type=str, default='data/model/nca_npy')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    train(args)


if __name__ == "__main__":
    main()
