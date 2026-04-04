"""
InfraMIND v3 — Result Visualizer
==================================

Generates matplotlib plots from experiment results and exports
JSON for the dashboard.

Usage:
    python scripts/visualize_results.py results/experiment_*.json
"""

import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Plot style
plt.rcParams.update({
    'figure.facecolor': '#0a0a1a',
    'axes.facecolor': '#0c0c1d',
    'axes.edgecolor': '#1e293b',
    'axes.labelcolor': '#94a3b8',
    'text.color': '#f1f5f9',
    'xtick.color': '#64748b',
    'ytick.color': '#64748b',
    'grid.color': '#1e293b',
    'grid.alpha': 0.4,
    'font.family': 'sans-serif',
    'font.size': 11,
    'legend.facecolor': '#0c0c1d',
    'legend.edgecolor': '#1e293b',
    'legend.labelcolor': '#94a3b8',
})

METHOD_COLORS = {
    'B1_Static': '#6b7280',
    'B2_Reactive': '#f59e0b',
    'B3_VanillaBO': '#06b6d4',
    'B4_StandardTuRBO': '#8b5cf6',
    'B5_InfraMINDv3': '#6366f1',
}

METHOD_LABELS = {
    'B1_Static': 'Static',
    'B2_Reactive': 'Reactive (HPA)',
    'B3_VanillaBO': 'Vanilla BO',
    'B4_StandardTuRBO': 'TuRBO',
    'B5_InfraMINDv3': 'InfraMIND v3',
}


def load_results(filepath):
    with open(filepath) as f:
        return json.load(f)


def plot_pareto(data, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for result in data['results']:
        method = result['method']
        obj = result['best_objective']
        color = METHOD_COLORS.get(method, '#94a3b8')
        label = METHOD_LABELS.get(method, method)
        marker = '*' if method == 'B5_InfraMINDv3' else 'o'
        size = 200 if method == 'B5_InfraMINDv3' else 100
        
        ax.scatter(obj['cost'], obj['p99'], c=color, s=size, marker=marker,
                  label=label, edgecolors='white', linewidths=0.5, zorder=5)
    
    sla = data['settings']['sla_target_p99_ms']
    ax.axhline(y=sla, color='#ef4444', linestyle='--', alpha=0.5, label=f'SLA Target ({sla}ms)')
    
    ax.set_xlabel('Cost ($)', fontsize=13)
    ax.set_ylabel('P99 Latency (ms)', fontsize=13)
    ax.set_title('Pareto Frontier — Cost vs Latency', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence(data, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    plotted = set()
    for result in data['results']:
        method = result['method']
        if method in plotted:
            continue
        plotted.add(method)
        
        if 'trajectory' not in result or not result['trajectory']:
            continue
        
        traj = result['trajectory']
        iterations = [t['iteration'] for t in traj]
        
        best_so_far = []
        best = float('inf')
        for t in traj:
            best = min(best, t['objective'])
            best_so_far.append(best)
        
        color = METHOD_COLORS.get(method, '#94a3b8')
        label = METHOD_LABELS.get(method, method)
        lw = 3 if method == 'B5_InfraMINDv3' else 1.5
        
        ax.plot(iterations, best_so_far, color=color, label=label, linewidth=lw)
    
    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Best Objective', fontsize=13)
    ax.set_title('Convergence — Best Objective vs Iteration', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_workload_comparison(data, output_dir):
    methods = sorted(set(r['method'] for r in data['results']))
    workloads = sorted(set(r['workload'] for r in data['results']))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(workloads))
    width = 0.15
    
    for i, method in enumerate(methods):
        values = []
        for wl in workloads:
            r = next((r for r in data['results'] if r['method'] == method and r['workload'] == wl), None)
            values.append(r['best_objective']['objective'] if r else 0)
        
        color = METHOD_COLORS.get(method, '#94a3b8')
        label = METHOD_LABELS.get(method, method)
        ax.bar(x + i * width, values, width, color=color, label=label, edgecolor='white', linewidth=0.3)
    
    ax.set_xlabel('Workload', fontsize=13)
    ax.set_ylabel('Objective', fontsize=13)
    ax.set_title('Workload Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([w.title() for w in workloads])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'workload_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="InfraMIND v3 — Visualize Results")
    parser.add_argument("input", type=str, help="Path to experiment results JSON")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    
    data = load_results(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Generating visualizations from {args.input}...")
    
    plot_pareto(data, output_dir)
    print("  ✅ Pareto frontier")
    
    plot_convergence(data, output_dir)
    print("  ✅ Convergence curves")
    
    plot_workload_comparison(data, output_dir)
    print("  ✅ Workload comparison")
    
    print(f"\n  Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
