"""
NCA Reference Data Generator & Comparison with StructGen-rs Data

This script:
1. Generates NCA (Neural Cellular Automata) reference data using JAX
2. Reads StructGen-rs parquet data
3. Compares statistical properties between the two datasets
"""

import sys
import os
import json
import gzip
import io
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.random import split
import flax.linen as nn
from flax.core import freeze, unfreeze
from einops import rearrange, repeat

import numpy as np
import pyarrow.parquet as pq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
#  NCA Implementation (standalone, from utils/nca.py)
# ============================================================

class NCANetwork(nn.Module):
    d_state: int = 16
    @nn.compact
    def __call__(self, x):
        x = jnp.pad(x, pad_width=1, mode='wrap')
        x = nn.Conv(features=4, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.Conv(features=16, kernel_size=(1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.d_state, kernel_size=(1, 1))(x)
        return x


class NCA:
    def __init__(self, grid_size=64, d_state=8, n_groups=1,
                 identity_bias=0., temperature=1.):
        self.grid_size = grid_size
        self.d_state = d_state
        self.n_groups = n_groups
        self.nca = NCANetwork(d_state=d_state * n_groups)
        self.identity_bias = identity_bias
        self.temperature = temperature

    def default_params(self, rng):
        params = dict()
        rng, _rng = split(rng)
        params['net_params'] = self.nca.init(
            _rng, jnp.zeros((self.grid_size, self.grid_size, self.d_state * self.n_groups))
        )
        rng, _rng = split(rng)
        params['init'] = jax.random.normal(_rng, (self.n_groups, self.d_state))
        return params

    def init_state(self, rng, params):
        init = repeat(
            jax.random.normal(rng, (self.n_groups, self.d_state)),
            "G D -> H W G D", H=self.grid_size, W=self.grid_size
        )
        state = jax.random.categorical(rng, init, axis=-1)
        return state

    def step_state(self, rng, state, params):
        state_oh = jax.nn.one_hot(state, self.d_state)
        state_oh_f = rearrange(state_oh, "H W G D -> H W (G D)")
        logits = self.nca.apply(params['net_params'], state_oh_f)
        logits = rearrange(logits, "H W (G D) -> H W G D", G=self.n_groups)
        next_state = jax.random.categorical(
            rng, (logits + state_oh * self.identity_bias) / self.temperature, axis=-1
        )
        return next_state


def rollout_simulation(rng, params, substrate, rollout_steps=256,
                       time_sampling='video', start_step=0, k_steps=1):
    """Rollout NCA simulation, returning state at each step."""
    s0 = substrate.init_state(rng, params)

    if time_sampling == 'video':
        def step_fn(state, _rng):
            next_state = substrate.step_state(_rng, state, params)
            return next_state, state
        _, state_vid = lax.scan(step_fn, s0, split(rng, start_step + rollout_steps))
        idx = jnp.arange(start_step, start_step + rollout_steps, k_steps)
        return state_vid[idx]
    else:
        raise ValueError(f"time_sampling {time_sampling} not supported in this script")


def generate_nca_dataset(seed, num_sims, grid=12, d_state=10, n_groups=1,
                         identity_bias=0., temperature=1., num_examples=10,
                         dT=1, start_step=0, rule_seeds=None):
    """Generate NCA dataset: returns array of shape (num_sims, num_examples, H, W, C)."""
    generator = NCA(grid_size=grid, d_state=d_state, n_groups=n_groups,
                    identity_bias=identity_bias, temperature=temperature)

    if rule_seeds is None:
        rule_seeds = jax.random.split(seed, num_sims)

    def rollout_fn(rng, task_seed):
        params = generator.default_params(task_seed)
        return rollout_simulation(rng, params, substrate=generator,
                                  rollout_steps=dT * num_examples,
                                  k_steps=dT, time_sampling='video',
                                  start_step=start_step)

    sim_data = jax.vmap(rollout_fn, in_axes=(0, 0))(
        split(seed, num_sims), rule_seeds
    )
    return sim_data


def gzip_complexity(byte_data: bytes) -> float:
    """Compute GZIP compression ratio."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(byte_data)
    compressed_size = len(buf.getvalue())
    original_size = len(byte_data)
    return compressed_size / original_size if original_size > 0 else 0


# ============================================================
#  StructGen-rs Parquet Reader
# ============================================================

def read_structgen_shard(path):
    """Read a StructGen-rs parquet shard and return structured data."""
    table = pq.read_table(path)
    df = table.to_pandas()
    return df


def decode_state_values(state_values_bytes, state_dim):
    """Decode binary state_values from StructGen-rs parquet.

    Each FrameState is encoded as 9 bytes:
      - 1 byte type tag: 0x01=Integer, 0x02=Float, 0x03=Bool
      - 8 bytes payload (little-endian)

    For CA data, all values are Bool (0x03), payload byte[0] = 0 or 1.
    """
    raw = np.frombuffer(state_values_bytes, dtype=np.uint8)
    n_values = len(raw) // 9
    values = np.empty(n_values, dtype=np.float64)

    for i in range(n_values):
        tag = raw[i * 9]
        payload = raw[i * 9 + 1 : i * 9 + 9]
        if tag == 0x03:  # Bool
            values[i] = payload[0]
        elif tag == 0x01:  # Integer
            values[i] = int.from_bytes(payload, byteorder='little', signed=True)
        elif tag == 0x02:  # Float
            values[i] = np.frombuffer(payload, dtype='<f8')[0]
        else:
            values[i] = 0

    if state_dim == 1:
        return values.astype(np.uint8)
    elif state_dim == 2:
        side = int(np.sqrt(n_values))
        return values.astype(np.uint8).reshape(side, side)
    elif state_dim == 3:
        side = round(n_values ** (1/3))
        return values.astype(np.uint8).reshape(side, side, side)
    return values.astype(np.uint8)


# ============================================================
#  Statistical Analysis Functions
# ============================================================

def compute_density(states):
    """Compute cell density (fraction of non-zero cells)."""
    return np.mean(states > 0)


def compute_entropy(states, num_bins=10):
    """Compute Shannon entropy of state distribution."""
    counts = np.bincount(states.flatten(), minlength=num_bins)[:num_bins]
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def compute_spatial_autocorrelation(state_2d):
    """Compute spatial autocorrelation (mean correlation of adjacent cells)."""
    if state_2d.ndim != 2:
        return 0.0
    h, w = state_2d.shape
    horiz_corr = np.mean(state_2d[:, :-1] == state_2d[:, 1:])
    vert_corr = np.mean(state_2d[:-1, :] == state_2d[1:, :])
    return (horiz_corr + vert_corr) / 2


def compute_temporal_change_rate(states_seq):
    """Compute rate of change between consecutive timesteps."""
    if len(states_seq) < 2:
        return 0.0
    changes = 0
    total = 0
    for i in range(len(states_seq) - 1):
        changes += np.sum(states_seq[i] != states_seq[i + 1])
        total += states_seq[i].size
    return changes / total if total > 0 else 0.0


def compute_pattern_diversity(states_list):
    """Compute number of unique patterns in a list of states."""
    unique_hashes = set()
    for s in states_list:
        unique_hashes.add(hash(s.tobytes()))
    return len(unique_hashes)


# ============================================================
#  Main Comparison
# ============================================================

def generate_nca_reference(output_dir, num_rules=20, num_sims_per_rule=10,
                           grid=12, d_state=10, num_steps=10):
    """Generate NCA reference data and save to disk."""
    print(f"\n{'='*60}")
    print(f"Generating NCA Reference Data")
    print(f"{'='*60}")
    print(f"  Grid size: {grid}x{grid}")
    print(f"  d_state: {d_state}")
    print(f"  num_rules: {num_rules}")
    print(f"  num_sims_per_rule: {num_sims_per_rule}")
    print(f"  num_steps: {num_steps}")

    seed = jax.random.PRNGKey(42)
    rule_seeds = jax.random.split(seed, num_rules)

    all_sims = []
    for i in range(num_rules):
        print(f"  Generating rule {i+1}/{num_rules}...")
        sim_seed = jax.random.PRNGKey(i + 1000)
        data = generate_nca_dataset(
            sim_seed, num_sims=num_sims_per_rule,
            grid=grid, d_state=d_state, n_groups=1,
            identity_bias=0., temperature=1.,
            num_examples=num_steps, dT=1, start_step=0,
            rule_seeds=jnp.array([rule_seeds[i]] * num_sims_per_rule)
        )
        all_sims.append(np.array(data))

    # Shape: (num_rules * num_sims_per_rule, num_steps, H, W, C)
    all_sims = np.concatenate(all_sims, axis=0)
    print(f"  Generated data shape: {all_sims.shape}")

    # Save as numpy
    np.save(os.path.join(output_dir, 'nca_reference_data.npy'), all_sims)
    print(f"  Saved to {output_dir}/nca_reference_data.npy")
    return all_sims


def analyze_nca_data(data):
    """Analyze NCA data and return statistics."""
    print(f"\n{'='*60}")
    print(f"NCA Data Analysis")
    print(f"{'='*60}")
    print(f"  Shape: {data.shape}")
    # data shape: (num_sims, num_steps, H, W, C)

    results = {
        'num_sims': data.shape[0],
        'num_steps': data.shape[1],
        'grid_h': data.shape[2],
        'grid_w': data.shape[3],
        'channels': data.shape[4] if data.ndim > 4 else 1,
    }

    # Density per timestep
    densities = []
    for t in range(data.shape[1]):
        step_data = data[:, t]
        if step_data.ndim > 3:
            step_data = step_data[..., 0]  # take first channel
        dens = [compute_density(step_data[s]) for s in range(min(100, step_data.shape[0]))]
        densities.append(np.mean(dens))

    results['density_mean'] = float(np.mean(densities))
    results['density_std'] = float(np.std(densities))
    results['density_per_step'] = [float(d) for d in densities]
    print(f"  Density: mean={results['density_mean']:.4f}, std={results['density_std']:.4f}")

    # Entropy
    entropies = []
    for t in range(data.shape[1]):
        step_data = data[:, t]
        if step_data.ndim > 3:
            step_data = step_data[..., 0]
        ent = [compute_entropy(step_data[s], num_bins=10) for s in range(min(100, step_data.shape[0]))]
        entropies.append(np.mean(ent))

    results['entropy_mean'] = float(np.mean(entropies))
    results['entropy_std'] = float(np.std(entropies))
    print(f"  Entropy: mean={results['entropy_mean']:.4f}, std={results['entropy_std']:.4f}")

    # Temporal change rate
    change_rates = []
    for s in range(min(100, data.shape[0])):
        sim = data[s]
        if sim.ndim > 3:
            sim = sim[..., 0]
        cr = compute_temporal_change_rate(sim)
        change_rates.append(cr)

    results['temporal_change_rate_mean'] = float(np.mean(change_rates))
    results['temporal_change_rate_std'] = float(np.std(change_rates))
    print(f"  Temporal change rate: mean={results['temporal_change_rate_mean']:.4f}, "
          f"std={results['temporal_change_rate_std']:.4f}")

    # Spatial autocorrelation
    autocorrs = []
    for s in range(min(100, data.shape[0])):
        for t in [0, data.shape[1] // 2, data.shape[1] - 1]:
            state = data[s, t]
            if state.ndim > 2:
                state = state[..., 0]
            if state.ndim == 2:
                ac = compute_spatial_autocorrelation(state)
                autocorrs.append(ac)

    results['spatial_autocorr_mean'] = float(np.mean(autocorrs)) if autocorrs else 0
    results['spatial_autocorr_std'] = float(np.std(autocorrs)) if autocorrs else 0
    print(f"  Spatial autocorrelation: mean={results['spatial_autocorr_mean']:.4f}, "
          f"std={results['spatial_autocorr_std']:.4f}")

    # GZIP complexity
    gzip_scores = []
    for s in range(min(50, data.shape[0])):
        for t in [0, data.shape[1] // 2, data.shape[1] - 1]:
            state = data[s, t]
            if state.ndim > 2:
                state = state[..., 0]
            byte_data = state.astype(np.uint8).tobytes()
            gc = gzip_complexity(byte_data)
            gzip_scores.append(gc)

    results['gzip_complexity_mean'] = float(np.mean(gzip_scores)) if gzip_scores else 0
    results['gzip_complexity_std'] = float(np.std(gzip_scores)) if gzip_scores else 0
    print(f"  GZIP complexity: mean={results['gzip_complexity_mean']:.4f}, "
          f"std={results['gzip_complexity_std']:.4f}")

    # Pattern diversity across rules
    final_states = data[:, -1]
    if final_states.ndim > 3:
        final_states = final_states[..., 0]
    diversity = compute_pattern_diversity(list(final_states))
    results['pattern_diversity'] = int(diversity)
    results['pattern_diversity_ratio'] = float(diversity / data.shape[0])
    print(f"  Pattern diversity: {diversity}/{data.shape[0]} "
          f"({results['pattern_diversity_ratio']:.2%})")

    return results


def analyze_structgen_data(parquet_dir, task_name, max_shards=3, max_samples=100):
    """Read and analyze StructGen-rs parquet data."""
    print(f"\n{'='*60}")
    print(f"StructGen-rs Data Analysis: {task_name}")
    print(f"{'='*60}")

    # Find parquet files for this task
    files = sorted([f for f in os.listdir(parquet_dir)
                    if f.startswith(task_name) and f.endswith('.parquet')
                    and not f.endswith('.tmp')])
    if not files:
        print(f"  No parquet files found for task {task_name}")
        return None

    print(f"  Found {len(files)} parquet shards")

    # Read first few shards
    all_data = []
    samples_read = 0
    for f in files[:max_shards]:
        path = os.path.join(parquet_dir, f)
        table = pq.read_table(path)
        df = table.to_pandas()
        all_data.append(df)
        samples_read += df['sample_id'].nunique()
        print(f"  Read {f}: {len(df)} rows, {df['sample_id'].nunique()} samples")
        if samples_read >= max_samples:
            break

    df = pd.concat(all_data, ignore_index=True) if len(all_data) > 1 else all_data[0]
    print(f"  Total rows: {len(df)}, unique samples: {df['sample_id'].nunique()}")

    # Determine actual dimensionality from task name
    # state_dim in parquet = number of cells (128, 1024, 512), NOT actual dim
    num_cells = int(df['state_dim'].iloc[0])
    if '1d' in task_name:
        actual_dim = 1
    elif '2d' in task_name:
        actual_dim = 2
    elif '3d' in task_name:
        actual_dim = 3
    else:
        actual_dim = 1 if num_cells < 200 else (2 if num_cells < 2000 else 3)
    print(f"  Num cells: {num_cells}, Actual dim: {actual_dim}")

    # Decode states
    results = {
        'task_name': task_name,
        'num_shards': len(files),
        'num_samples': int(df['sample_id'].nunique()),
        'state_dim': actual_dim,
        'num_cells': num_cells,
    }

    # Get unique sample_ids and step range
    unique_sids = df['sample_id'].unique()[:max_samples]
    max_step = df['step_index'].max()
    results['max_step'] = int(max_step)

    # Decode state values for analysis
    densities = []
    entropies = []
    autocorrs = []
    change_rates = []
    gzip_scores = []

    for sid in unique_sids:
        sample_df = df[df['sample_id'] == sid].sort_values('step_index')
        states = []
        for _, row in sample_df.iterrows():
            state = decode_state_values(row['state_values'], actual_dim)
            states.append(state)

        if not states:
            continue

        # Density
        dens = [compute_density(s) for s in states]
        densities.append(np.mean(dens))

        # Entropy
        ent = [compute_entropy(s, num_bins=max(2, 10)) for s in states]
        entropies.append(np.mean(ent))

        # Spatial autocorrelation (2D only)
        if actual_dim == 2:
            for s in states[:3]:
                ac = compute_spatial_autocorrelation(s)
                autocorrs.append(ac)

        # Temporal change rate
        cr = compute_temporal_change_rate(states)
        change_rates.append(cr)

        # GZIP complexity
        for s in states[:3]:
            byte_data = s.astype(np.uint8).tobytes()
            gc = gzip_complexity(byte_data)
            gzip_scores.append(gc)

    results['density_mean'] = float(np.mean(densities)) if densities else 0
    results['density_std'] = float(np.std(densities)) if densities else 0
    results['entropy_mean'] = float(np.mean(entropies)) if entropies else 0
    results['entropy_std'] = float(np.std(entropies)) if entropies else 0
    results['spatial_autocorr_mean'] = float(np.mean(autocorrs)) if autocorrs else 0
    results['spatial_autocorr_std'] = float(np.std(autocorrs)) if autocorrs else 0
    results['temporal_change_rate_mean'] = float(np.mean(change_rates)) if change_rates else 0
    results['temporal_change_rate_std'] = float(np.std(change_rates)) if change_rates else 0
    results['gzip_complexity_mean'] = float(np.mean(gzip_scores)) if gzip_scores else 0
    results['gzip_complexity_std'] = float(np.std(gzip_scores)) if gzip_scores else 0

    print(f"  Density: mean={results['density_mean']:.4f}, std={results['density_std']:.4f}")
    print(f"  Entropy: mean={results['entropy_mean']:.4f}, std={results['entropy_std']:.4f}")
    print(f"  Spatial autocorrelation: mean={results['spatial_autocorr_mean']:.4f}")
    print(f"  Temporal change rate: mean={results['temporal_change_rate_mean']:.4f}")
    print(f"  GZIP complexity: mean={results['gzip_complexity_mean']:.4f}")

    return results


def generate_comparison_report(nca_results, structgen_results_list, output_dir):
    """Generate a comparison report and plots."""
    print(f"\n{'='*60}")
    print(f"COMPARISON REPORT: NCA vs StructGen-rs")
    print(f"{'='*60}")

    report = {
        'timestamp': datetime.now().isoformat(),
        'nca': nca_results,
        'structgen': structgen_results_list,
        'comparison': {}
    }

    # Compare each StructGen task with NCA
    for sg in structgen_results_list:
        if sg is None:
            continue
        task = sg['task_name']
        print(f"\n--- {task} vs NCA ---")

        comparison = {}

        # Density comparison
        nca_dens = nca_results.get('density_mean', 0)
        sg_dens = sg.get('density_mean', 0)
        comparison['density_diff'] = float(abs(nca_dens - sg_dens))
        comparison['density_ratio'] = float(nca_dens / sg_dens) if sg_dens > 0 else float('inf')
        print(f"  Density: NCA={nca_dens:.4f}, SG={sg_dens:.4f}, "
              f"diff={comparison['density_diff']:.4f}")

        # Entropy comparison
        nca_ent = nca_results.get('entropy_mean', 0)
        sg_ent = sg.get('entropy_mean', 0)
        comparison['entropy_diff'] = float(abs(nca_ent - sg_ent))
        print(f"  Entropy: NCA={nca_ent:.4f}, SG={sg_ent:.4f}, "
              f"diff={comparison['entropy_diff']:.4f}")

        # GZIP complexity comparison
        nca_gz = nca_results.get('gzip_complexity_mean', 0)
        sg_gz = sg.get('gzip_complexity_mean', 0)
        comparison['gzip_diff'] = float(abs(nca_gz - sg_gz))
        print(f"  GZIP complexity: NCA={nca_gz:.4f}, SG={sg_gz:.4f}, "
              f"diff={comparison['gzip_diff']:.4f}")

        # Temporal change rate
        nca_cr = nca_results.get('temporal_change_rate_mean', 0)
        sg_cr = sg.get('temporal_change_rate_mean', 0)
        comparison['change_rate_diff'] = float(abs(nca_cr - sg_cr))
        print(f"  Temporal change rate: NCA={nca_cr:.4f}, SG={sg_cr:.4f}, "
              f"diff={comparison['change_rate_diff']:.4f}")

        report['comparison'][task] = comparison

    # Save report
    report_path = os.path.join(output_dir, 'comparison_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Report saved to {report_path}")

    # Generate comparison plot
    generate_comparison_plots(nca_results, structgen_results_list, output_dir)

    return report


def generate_comparison_plots(nca_results, structgen_results_list, output_dir):
    """Generate comparison visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NCA vs StructGen-rs: Statistical Comparison', fontsize=14)

    metrics = [
        ('density_mean', 'density_std', 'Cell Density'),
        ('entropy_mean', 'entropy_std', 'Shannon Entropy'),
        ('gzip_complexity_mean', 'gzip_complexity_std', 'GZIP Complexity'),
        ('temporal_change_rate_mean', 'temporal_change_rate_std', 'Temporal Change Rate'),
    ]

    for idx, (mean_key, std_key, title) in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]

        labels = ['NCA']
        means = [nca_results.get(mean_key, 0)]
        stds = [nca_results.get(std_key, 0)]

        for sg in structgen_results_list:
            if sg is None:
                continue
            name = sg['task_name']
            if '1d' in name:
                short = '1D-Rule30'
            elif '2d' in name:
                short = '2D-Life'
            elif '3d' in name:
                short = '3D-Life'
            else:
                short = name
            labels.append(short)
            means.append(sg.get(mean_key, 0))
            stds.append(sg.get(std_key, 0))

        x = range(len(labels))
        ax.bar(x, means, yerr=stds, capsize=5, color=['#2196F3'] + ['#FF9800'] * (len(labels) - 1),
               alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(title)
        ax.set_ylabel('Value')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison plot saved to {plot_path}")


def generate_sample_visualizations(nca_data, parquet_dir, output_dir):
    """Generate sample visualizations for both datasets."""
    # NCA samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('NCA Sample Evolution (5 rules, first 2 timesteps)', fontsize=12)
    for i in range(5):
        rule_idx = i * (nca_data.shape[0] // 5)
        for t in range(2):
            state = nca_data[rule_idx, t]
            if state.ndim > 2:
                state = state[..., 0]
            ax = axes[t][i]
            ax.imshow(state, cmap='tab10', vmin=0, vmax=9)
            ax.set_title(f'R{i} T{t}')
            ax.axis('off')
    plt.tight_layout()
    nca_viz_path = os.path.join(output_dir, 'nca_samples.png')
    plt.savefig(nca_viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCA visualization saved to {nca_viz_path}")

    # StructGen 2D samples
    files_2d = sorted([f for f in os.listdir(parquet_dir)
                       if f.startswith('ca_2d_life') and f.endswith('.parquet')
                       and not f.endswith('.tmp')])
    if files_2d:
        df = pq.read_table(os.path.join(parquet_dir, files_2d[0])).to_pandas()
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('StructGen 2D Life Sample Evolution', fontsize=12)
        for i in range(5):
            sample_ids = df['sample_id'].unique()
            sid = sample_ids[i * (len(sample_ids) // 5)] if len(sample_ids) >= 5 else sample_ids[0]
            sample_df = df[df['sample_id'] == sid].sort_values('step_index')
            steps = sample_df['step_index'].values
            for t_idx, t in enumerate([0, min(len(steps) - 1, 10)]):
                row = sample_df[sample_df['step_index'] == steps[t]].iloc[0]
                state = decode_state_values(row['state_values'], 2)
                ax = axes[t_idx][i]
                ax.imshow(state, cmap='binary')
                ax.set_title(f'S{sid} T{steps[t]}')
                ax.axis('off')
        plt.tight_layout()
        sg_viz_path = os.path.join(output_dir, 'structgen_2d_samples.png')
        plt.savefig(sg_viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  StructGen 2D visualization saved to {sg_viz_path}")

    # StructGen 1D samples
    files_1d = sorted([f for f in os.listdir(parquet_dir)
                       if f.startswith('ca_1d_rule30') and f.endswith('.parquet')
                       and not f.endswith('.tmp')])
    if files_1d:
        df = pq.read_table(os.path.join(parquet_dir, files_1d[0])).to_pandas()
        fig, axes = plt.subplots(5, 1, figsize=(15, 8))
        fig.suptitle('StructGen 1D Rule30 Space-Time Diagrams', fontsize=12)
        sample_ids = df['sample_id'].unique()[:5]
        for i, sid in enumerate(sample_ids):
            sample_df = df[df['sample_id'] == sid].sort_values('step_index')
            spacetime = np.array([
                decode_state_values(row['state_values'], 1)
                for _, row in sample_df.iterrows()
            ])
            axes[i].imshow(spacetime, cmap='binary', aspect='auto')
            axes[i].set_ylabel(f'S{sid}')
            axes[i].set_xlabel('Cell')
        plt.tight_layout()
        sg_1d_path = os.path.join(output_dir, 'structgen_1d_samples.png')
        plt.savefig(sg_1d_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  StructGen 1D visualization saved to {sg_1d_path}")


# ============================================================
#  Main
# ============================================================

if __name__ == '__main__':
    import pandas as pd

    PARQUET_DIR = r'F:\RustProjects\StructGen-rs\output_ca'
    OUTPUT_DIR = r'F:\RustProjects\StructGen-rs\comparison_output'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Generate NCA reference data
    print("\n" + "=" * 60)
    print("STEP 1: Generating NCA Reference Data")
    print("=" * 60)
    nca_data = generate_nca_reference(
        OUTPUT_DIR,
        num_rules=20,
        num_sims_per_rule=5,
        grid=12,
        d_state=10,
        num_steps=10
    )

    # Step 2: Analyze NCA data
    print("\n" + "=" * 60)
    print("STEP 2: Analyzing NCA Data")
    print("=" * 60)
    nca_results = analyze_nca_data(nca_data)

    # Step 3: Analyze StructGen-rs data
    print("\n" + "=" * 60)
    print("STEP 3: Analyzing StructGen-rs Data")
    print("=" * 60)
    structgen_results = []
    for task_name in ['ca_1d_rule30', 'ca_2d_life', 'ca_3d_life']:
        result = analyze_structgen_data(PARQUET_DIR, task_name, max_shards=2, max_samples=50)
        if result:
            structgen_results.append(result)

    # Step 4: Generate comparison report
    print("\n" + "=" * 60)
    print("STEP 4: Generating Comparison Report")
    print("=" * 60)
    report = generate_comparison_report(nca_results, structgen_results, OUTPUT_DIR)

    # Step 5: Generate visualizations
    print("\n" + "=" * 60)
    print("STEP 5: Generating Visualizations")
    print("=" * 60)
    generate_sample_visualizations(nca_data, PARQUET_DIR, OUTPUT_DIR)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print("\nKey Differences Found:")
    print("-" * 40)

    for sg in structgen_results:
        if sg is None:
            continue
        task = sg['task_name']
        comp = report['comparison'].get(task, {})
        print(f"\n  {task}:")
        print(f"    State dimensionality: {sg['state_dim']}D")
        print(f"    Density difference: {comp.get('density_diff', 0):.4f}")
        print(f"    Entropy difference: {comp.get('entropy_diff', 0):.4f}")
        print(f"    GZIP complexity diff: {comp.get('gzip_diff', 0):.4f}")
        print(f"    Change rate diff: {comp.get('change_rate_diff', 0):.4f}")

    print(f"\n  NCA characteristics:")
    print(f"    Uses neural network rules (randomly initialized)")
    print(f"    d_state=10 (10 possible cell values)")
    print(f"    Grid 12x12 with patch tokenization")
    print(f"    Each rule produces unique dynamics")

    print(f"\n  StructGen-rs characteristics:")
    print(f"    Uses fixed CA rules (Rule 30, B3/S23, B567/S56)")
    print(f"    Binary state (0/1)")
    print(f"    Larger grids (128 for 1D, 32x32 for 2D, 8x8x8 for 3D)")
    print(f"    Same rule for all samples (diversity from initial conditions only)")

    print(f"\n\nAll outputs saved to: {OUTPUT_DIR}")
