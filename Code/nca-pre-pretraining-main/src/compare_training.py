"""
对比训练脚本 - 比较不同生成器数据的训练效果
包含混合数据训练模式，支持三种课程学习策略：
  random     - 完全随机混合
  easy_hard  - 由易到难（先简单规则，后复杂规则）
  hard_easy  - 由难到易（先复杂规则，后简单规则）
"""

import os, sys, argparse, glob, math, logging, random
from typing import Optional, Dict, List

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger('compare_train')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 难度排序（基于单一数据训练的初始 loss）
DATA_DIFFICULTY = {
    'ca_rule110': 0.6683,
    'ca2d_life':  0.7158,
    'ca2d_maze':  0.7397,
    'ca_rule30':  0.7773,
    'nca2d':      2.2713,
    'ca2d_cyclic':2.7861,
}


class SimpleTokenizer:
    def __init__(self, num_colors: int, start_offset: int = 0):
        self.num_colors = num_colors
        self.start_offset = start_offset
        self.start_tk = start_offset + num_colors
        self.end_tk = start_offset + num_colors + 1
        self.vocab_size = num_colors + 2

    def encode_task(self, grid: np.ndarray):
        B, N = grid.shape[0], grid.shape[1]
        if grid.ndim == 5:
            grid = np.argmax(grid, axis=-1)
        H, W = grid.shape[2], grid.shape[3]
        grid = grid.astype(np.int64).reshape(B, N, H * W)
        start = np.full((B, N, 1), self.start_tk, dtype=np.int64)
        end = np.full((B, N, 1), self.end_tk, dtype=np.int64)
        tokens = np.concatenate([start, grid, end], axis=-1)
        target = tokens.copy()
        target[:, :, 0] = -100
        target[:, :, -1] = -100
        return (torch.tensor(tokens.reshape(B, -1), dtype=torch.long),
                torch.tensor(target.reshape(B, -1), dtype=torch.long))


class NpyDataset(Dataset):
    def __init__(self, data_dir, pattern, max_samples=None, max_seq_len=512, start_offset=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not self.files:
            raise ValueError(f"未找到匹配 {pattern} 的文件")
        all_data = []
        total = 0
        for f in self.files:
            d = np.load(f)
            all_data.append(d)
            total += d.shape[0]
            if max_samples and total >= max_samples:
                break
        self.data = np.concatenate(all_data, axis=0)
        if max_samples:
            self.data = self.data[:max_samples]
        self.T = self.data.shape[1]
        if self.data.ndim == 5:
            self.H, self.W, self.C = self.data.shape[2], self.data.shape[3], self.data.shape[4]
            self.num_colors = self.C
        else:
            self.H, self.W = self.data.shape[2], self.data.shape[3]
            self.num_colors = int(self.data.max()) + 1
        self.tokenizer = SimpleTokenizer(self.num_colors, start_offset)
        self.seq, self.targets = self.tokenizer.encode_task(self.data)
        self.grid_len = self.H * self.W + 2
        self.seq_len = self.T * self.grid_len

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq, targets = self.seq[idx].clone(), self.targets[idx].clone()
        seq_out, targets_out = seq[:-1], targets[1:]
        if seq_out.shape[0] > self.max_seq_len:
            seq_out, targets_out = seq_out[:self.max_seq_len], targets_out[:self.max_seq_len]
        elif seq_out.shape[0] < self.max_seq_len:
            pad = self.max_seq_len - seq_out.shape[0]
            seq_out = torch.cat([seq_out, torch.full((pad,), -100, dtype=seq_out.dtype)])
            targets_out = torch.cat([targets_out, torch.full((pad,), -100, dtype=targets_out.dtype)])
        return seq_out.float(), targets_out.long()


class MixedDataset(Dataset):
    """混合数据集，支持三种课程学习模式"""

    def __init__(self, data_dir, patterns, max_samples_per_type,
                 max_seq_len=512, curriculum='random'):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.curriculum = curriculum
        self.sub_datasets = []
        self.vocab_info = {}

        # 扫描 num_colors
        type_info = {}
        for dt_name, pattern in patterns.items():
            try:
                files = sorted(glob.glob(os.path.join(data_dir, pattern)))
                if not files:
                    continue
                sample = np.load(files[0])
                nc = sample.shape[-1] if sample.ndim == 5 else int(sample.max()) + 1
                type_info[dt_name] = {'pattern': pattern, 'num_colors': nc}
            except Exception as e:
                log.warning(f"扫描 {dt_name} 失败: {e}")

        if not type_info:
            raise ValueError("没有找到任何可用的数据类型")

        # 按难度排序数据类型名称
        if curriculum == 'easy_hard':
            ordered_names = sorted(type_info.keys(),
                                   key=lambda n: DATA_DIFFICULTY.get(n, 999))
        elif curriculum == 'hard_easy':
            ordered_names = sorted(type_info.keys(),
                                   key=lambda n: DATA_DIFFICULTY.get(n, 999), reverse=True)
        else:  # random
            ordered_names = sorted(type_info.keys())  # 字母顺序，内部 shuffle

        # 计算统一词汇表偏移
        offset = 0
        for dt_name in ordered_names:
            nc = type_info[dt_name]['num_colors']
            self.vocab_info[dt_name] = {
                'num_colors': nc, 'start_offset': offset,
                'token_range': (offset, offset + nc),
                'start_tk': offset + nc, 'end_tk': offset + nc + 1,
                'difficulty': DATA_DIFFICULTY.get(dt_name, 1.0),
            }
            offset += nc + 2
        self.unified_vocab_size = offset

        # 按排序后的顺序加载数据集
        for dt_name in ordered_names:
            info = type_info[dt_name]
            vi = self.vocab_info[dt_name]
            try:
                ds = NpyDataset(data_dir, info['pattern'],
                                max_samples=max_samples_per_type,
                                max_seq_len=max_seq_len,
                                start_offset=vi['start_offset'])
                self.sub_datasets.append((dt_name, ds))
                log.info(f"[mixed/{curriculum}] {dt_name:15s}: {len(ds)} samples, "
                         f"colors={vi['num_colors']}, difficulty={vi['difficulty']:.3f}")
            except Exception as e:
                log.warning(f"[mixed] 加载 {dt_name} 失败: {e}")

        # 构建 flat_indices
        self.flat_indices = []
        for dt_idx, (dt_name, ds) in enumerate(self.sub_datasets):
            for i in range(len(ds)):
                self.flat_indices.append((dt_idx, i))

        if curriculum == 'random':
            random.shuffle(self.flat_indices)
        # easy_hard: 保持自然顺序（简单数据在前）
        # hard_easy: 保持自然顺序（困难数据在前，因为 sub_datasets 已按 reverse 排序）

        self.grid_len = max(ds.grid_len for _, ds in self.sub_datasets)
        self.seq_len = max(ds.seq_len for _, ds in self.sub_datasets)
        total_items = sum(ds.data.size for _, ds in self.sub_datasets) / (1024**3)

        log.info(f"[mixed/{curriculum}] total={len(self)}, "
                 f"unified_vocab={self.unified_vocab_size}, "
                 f"seq_len={self.seq_len}, grid_len={self.grid_len}")
        log.info(f"[mixed/{curriculum}] order: {' -> '.join(n for n, _ in self.sub_datasets)}")

    def __len__(self):
        return len(self.flat_indices)

    def __getitem__(self, idx):
        dt_idx, sample_idx = self.flat_indices[idx]
        _, ds = self.sub_datasets[dt_idx]
        return ds[sample_idx]


# ===== 模型 ===== #
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

class Attention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head, self.head_dim = n_head, dim // n_head
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(dim, dim, bias=False) for _ in range(4))
    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask
        return self.wo((F.softmax(attn, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1, self.w2 = nn.Linear(dim, hidden_dim, bias=False), nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.attn = Attention(dim, n_head)
        self.ffn = FeedForward(dim, dim * 4)
        self.attn_norm, self.ffn_norm = RMSNorm(dim), RMSNorm(dim)
    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.ffn(self.ffn_norm(x))

class LLaMA(nn.Module):
    def __init__(self, vocab_size, dim, n_layer, n_head, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_head) for _ in range(n_layer)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.seq_len = seq_len
    def forward(self, x, mask=None):
        B, T = x.shape
        x = torch.clamp(x, min=0).long()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h, mask)
        return self.head(self.norm(h))

def create_attention_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).masked_fill(
        torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 1, float('-inf')).unsqueeze(0).unsqueeze(0)


# ===== 训练 ===== #

def _train_loop(model, loader, device, seq_len, args, desc=""):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda s: s / warmup_steps if s < warmup_steps else max(0, 1.0 - (s - warmup_steps) / (total_steps - warmup_steps)))
    criterion = CrossEntropyLoss(ignore_index=-100)
    attn_mask = create_attention_mask(seq_len).to(device)
    loss_history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        pbar = tqdm(loader, desc=f"{desc} E{epoch+1}/{args.epochs}", leave=False)
        for seq, targets in pbar:
            seq, targets = seq.to(device), targets.to(device)
            B, T = seq.shape
            mask = attn_mask.expand(B, -1, -1, -1)
            logits = model(seq, mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        avg = epoch_loss / n if n > 0 else 0
        loss_history.append(avg)
        log.info(f"{desc} Epoch {epoch+1}: loss={avg:.4f}")
    return loss_history


def train_mixed(args, patterns, curriculum):
    """训练混合数据（指定课程模式）"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.save_dir, f"mixed_{curriculum}")
    os.makedirs(save_dir, exist_ok=True)

    n_types = len(patterns)
    samples_per_type = max(args.train_samples // n_types, 100)
    log.info(f"[mixed/{curriculum}] 加载数据集 (每种 {samples_per_type} 样本)...")

    try:
        dataset = MixedDataset(args.data_dir, patterns, samples_per_type,
                               max_seq_len=args.seq_len, curriculum=curriculum)
    except ValueError as e:
        log.warning(f"[mixed/{curriculum}] 无法加载: {e}")
        return {"data_type": f"mixed_{curriculum}", "loss_history": [], "error": str(e)}

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)
    # 注意：shuffle=False，因为课程顺序由 flat_indices 控制

    model = LLaMA(dataset.unified_vocab_size, args.n_embd, args.n_layer,
                  args.n_head, args.seq_len).to(device)
    log.info(f"[mixed/{curriculum}] params={sum(p.numel() for p in model.parameters()):,}, "
             f"vocab={dataset.unified_vocab_size}, seq_len={dataset.seq_len}")

    loss_history = _train_loop(model, loader, device, args.seq_len, args,
                               desc=f"[mixed/{curriculum}]")

    torch.save({'loss_history': loss_history, 'model': model.state_dict(),
                'vocab_info': dataset.vocab_info, 'curriculum': curriculum},
               os.path.join(save_dir, 'final_model.pt'))

    return {"data_type": f"mixed_{curriculum}", "loss_history": loss_history,
            "curriculum": curriculum,
            "final_loss": loss_history[-1] if loss_history else None}


# ===== 主函数 ===== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='F:/RustProjects/StructGen-rs/output/batch_generation')
    parser.add_argument('--save_dir', default='F:/RustProjects/StructGen-rs/Code/nca-pre-pretraining-main/data/model/compare')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--train_samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--data_types', nargs='+', default=[
        'nca2d', 'ca2d_life', 'ca2d_maze', 'ca2d_cyclic', 'ca_rule30', 'ca_rule110'
    ])
    parser.add_argument('--curriculum_mode', type=str, default='all',
                        choices=['random', 'easy_hard', 'hard_easy', 'all'],
                        help='课程学习模式: random=随机, easy_hard=由易到难, hard_easy=由难到易, all=全部三种')
    args = parser.parse_args()

    patterns = {
        'nca2d': 'nca2d_batch_*.npy',
        'ca2d_life': 'ca2d_life_batch_*.npy',
        'ca2d_maze': 'ca2d_maze_batch_*.npy',
        'ca2d_cyclic': 'ca2d_cyclic_batch_*.npy',
        'ca_rule30': 'ca_rule30_batch_*.npy',
        'ca_rule110': 'ca_rule110_batch_*.npy',
    }
    mixed_patterns = {dt: patterns[dt] for dt in args.data_types if dt in patterns}

    modes = ['random', 'easy_hard', 'hard_easy'] if args.curriculum_mode == 'all' else [args.curriculum_mode]

    log.info("=" * 60)
    log.info(f"课程学习对比: {modes}")
    log.info(f"样本/类型: {max(args.train_samples // len(mixed_patterns), 100)}")
    log.info(f"总轮次: {args.epochs}")
    log.info("=" * 60)

    results = {}
    for mode in modes:
        log.info(f"\n{'='*50}\n课程模式: {mode}\n{'='*50}")
        results[mode] = train_mixed(args, mixed_patterns, mode)
        if results[mode].get("loss_history"):
            log.info(f"[{mode}] final_loss={results[mode]['final_loss']:.4f}")

    # 绘图
    plot_curriculum_results(results, args.save_dir)

    # 汇总
    log.info("\n" + "=" * 70)
    log.info("课程学习结果汇总")
    log.info("=" * 70)
    for mode, r in results.items():
        if r.get("loss_history"):
            init, final = r['loss_history'][0], r['loss_history'][-1]
            drop_pct = (init - final) / init * 100 if init > 0 else 0
            log.info(f"{mode:20s}: {init:.4f} -> {final:.4f} (下降 {drop_pct:.1f}%)")

    # 分析
    if len(results) >= 2:
        analyze_curriculum(results)


def analyze_curriculum(results):
    """分析课程学习效果差异"""
    log.info("\n" + "=" * 70)
    log.info("课程学习效果分析")
    log.info("=" * 70)

    stats = {}
    for mode, r in results.items():
        if r.get("loss_history"):
            h = r["loss_history"]
            stats[mode] = {
                'init': h[0],
                'final': h[-1],
                'drop_pct': (h[0] - h[-1]) / h[0] * 100,
                'min': min(h),
                'min_epoch': h.index(min(h)) + 1,
                'first_5': sum(h[:5]) / 5,
                'last_5': sum(h[-5:]) / 5,
            }

    # 最快收敛
    best = max(stats.items(), key=lambda x: x[1]['drop_pct'])
    best_init = min(stats.items(), key=lambda x: x[1]['init'])
    best_final = min(stats.items(), key=lambda x: x[1]['final'])
    best_min = min(stats.items(), key=lambda x: x[1]['min'])

    log.info(f"最快 loss 下降:    {best[0]} ({best[1]['drop_pct']:.1f}%)")
    log.info(f"最低初始 loss:     {best_init[0]} ({best_init[1]['init']:.4f})")
    log.info(f"最低最终 loss:     {best_final[0]} ({best_final[1]['final']:.4f})")
    log.info(f"最低 loss (全局):  {best_min[0]} ({best_min[1]['min']:.4f} @ epoch {best_min[1]['min_epoch']})")

    # 前5轮 vs 后5轮
    log.info("\n前5轮 vs 后5轮 平均 loss:")
    for mode, s in stats.items():
        log.info(f"  {mode:20s}: {s['first_5']:.4f} -> {s['last_5']:.4f} "
                 f"({(s['first_5'] - s['last_5']) / s['first_5'] * 100:.1f}%)")

    # 结论
    modes_list = list(stats.keys())
    if len(modes_list) >= 2:
        a, b = modes_list[0], modes_list[1]
        if stats[a]['final'] < stats[b]['final']:
            log.info(f"\n结论: {a} > {b} (最终 loss 更低)")
        else:
            log.info(f"\n结论: {b} > {a} (最终 loss 更低)")

    if len(modes_list) >= 3:
        ranks = sorted(modes_list, key=lambda m: stats[m]['final'])
        log.info(f"课程模式排名 (按最终 loss): {' > '.join(ranks)}")


def plot_curriculum_results(results, save_dir):
    """绘制课程学习对比图"""
    mode_colors = {'random': '#4472C4', 'easy_hard': '#ED7D31', 'hard_easy': '#70AD47'}
    mode_styles = {'random': '-', 'easy_hard': '--', 'hard_easy': ':'}
    mode_markers = {'random': 'o', 'easy_hard': 's', 'hard_easy': '^'}
    mode_labels = {'random': '随机混合', 'easy_hard': '由易到难', 'hard_easy': '由难到易'}

    # 图1: loss 曲线
    fig, ax = plt.subplots(figsize=(14, 8))
    for mode, r in results.items():
        if not r.get("loss_history"):
            continue
        h = r["loss_history"]
        label = mode_labels.get(r.get('curriculum', mode), mode)
        ax.plot(range(1, len(h)+1), h,
                label=label,
                color=mode_colors.get(mode, 'gray'),
                linewidth=2.5,
                linestyle=mode_styles.get(mode, '-'),
                marker=mode_markers.get(mode, 'o'),
                markersize=4,
                markevery=3)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Curriculum Learning: Random vs Easy-Hard vs Hard-Easy (Mixed CA Data)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curriculum_loss_comparison.png"), dpi=150)
    plt.close()

    # 图2: 下降百分比 + 子图展示前5/后5轮 loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1: 柱状图
    labels, drops, colors = [], [], []
    for mode, r in results.items():
        if r.get("loss_history") and len(r["loss_history"]) > 0:
            h = r["loss_history"]
            drops.append((h[0] - h[-1]) / h[0] * 100)
            labels.append(mode_labels.get(r.get('curriculum', mode), mode))
            colors.append(mode_colors.get(mode, 'gray'))
    if drops:
        bars = ax1.bar(labels, drops, color=colors)
        for bar, d in zip(bars, drops):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{d:.1f}%', ha='center', fontsize=10)
    ax1.set_title('Loss Drop %', fontsize=13)
    ax1.tick_params(axis='x', rotation=20)

    # 子图2: early vs late loss
    x = np.arange(len(results))
    width = 0.25
    for i, (mode, r) in enumerate(results.items()):
        if not r.get("loss_history"):
            continue
        h = r["loss_history"]
        label = mode_labels.get(r.get('curriculum', mode), mode)
        early = sum(h[:5]) / 5 if len(h) >= 5 else h[0]
        late = sum(h[-5:]) / 5 if len(h) >= 10 else h[-1]
        ax2.bar(i - width/2, early, width, label='Epochs 1-5' if i == 0 else '',
                color=mode_colors.get(mode, 'gray'), alpha=0.7)
        ax2.bar(i + width/2, late, width, label='Epochs 26-30' if i == 0 else '',
                color=mode_colors.get(mode, 'gray'), alpha=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([mode_labels.get(r.get('curriculum', m), m) for m, r in results.items()],
                        rotation=20)
    ax2.set_title('Early vs Late Training Loss', fontsize=13)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curriculum_drop_comparison.png"), dpi=150)
    plt.close()
    log.info(f"课程学习对比图已保存至: {save_dir}")


if __name__ == "__main__":
    main()
