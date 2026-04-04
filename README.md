# 🧠 InfraMIND v3

**Workload-Aware, Structure-Learning, Stability-Optimized Infrastructure Intelligence under High-Dimensional Constraints**

> *We argue that infrastructure optimization is fundamentally a structured, workload-conditioned problem rather than a flat black-box search task.*

---

## 🔬 Novel Contributions

| # | Contribution | Key Innovation |
|---|-------------|----------------|
| **C1** | Workload-Sensitive Trust Region Adaptation | Trust region scales inversely with workload burstiness |
| **C2** | Data-Driven Structure Learning | Sensitivity-based spectral clustering for parameter grouping |
| **C3** | Stability-Aware Objective | CV² latency variance penalty beyond threshold-only optimization |

**Supporting:** Workload-conditioned GP (C4) · Multi-service DAG simulation (C5) · Cross-workload generalization (C6)

---

## 🏗️ Architecture

```
config → workload_generator → embedder → simulator → metrics → structure_learner → param_mapper → adaptive_turbo → (repeat)
```

```
InfraMIND/
├── config/                # YAML config + settings loader
├── workloads/             # Trace generation (steady/diurnal/bursty)
├── embeddings/            # 5D workload feature extraction
├── simulator/             # SimPy DAG simulation engine
├── metrics/               # Stability-aware objective (C3)
├── structure_learning/    # Sensitivity analysis + spectral clustering (C2)
├── optimizer/             # Adaptive TuRBO (C1) + GP surrogate + baselines
├── experiments/           # Runner, ablation study, generalization test
├── dashboard/             # Premium dark-mode results dashboard
├── tests/                 # Unit test suite
└── scripts/               # Execution scripts
```

---

## ⚡ Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Single Optimization (Smoke Test)

```bash
python scripts/run_single_optimization.py --n-iter 10 --workload bursty
```

### 3. Run Full Experiment Suite

```bash
python scripts/run_full_experiment.py --n-iter 30 --n-trials 2
```

### 4. View Dashboard

Open `dashboard/index.html` in any browser, then either:
- Click **"Load Results"** to load `results/experiment_*.json`
- Click **"Demo Data"** for synthetic visualization

### 5. Generate Plots

```bash
python scripts/visualize_results.py results/experiment_*.json
```

---

## 🧪 Experimental Design

### Baselines
| ID | Method | Description |
|----|--------|-------------|
| B1 | Static | Fixed midpoint provisioning |
| B2 | Reactive | Threshold-based HPA-style auto-scaling |
| B3 | Vanilla BO | GP-EI without workload conditioning |
| B4 | Standard TuRBO | Trust region BO without adaptation |
| B5 | **InfraMIND v3** | Full system with all contributions |

### Ablation Study
Remove one component at a time: `-embedding`, `-structure`, `-adaptive_tr`, `-stability`

### Generalization Test
- **Train:** Steady + Diurnal workloads
- **Test:** Bursty (unseen) — evaluates zero-shot transfer

---

## 📊 Objective Function

$$\min_{\theta \in \Theta} \; \mathbb{E}_{W \sim \mathcal{D}} \left[ \text{Cost}(\theta, W) + \lambda_1 \cdot \text{SLA}_{\text{viol}} + \lambda_2 \cdot \text{CV}^2(\text{Latency}) \right]$$

Where:
- **Cost** = Σ replicas × CPU_factor × rate × time
- **SLA_viol** = fraction of requests exceeding P99 target
- **CV²** = (σ/μ)² — scale-invariant stability measure

---

## 🔧 Key Configuration

Edit `config/default_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sla_target_p99_ms` | 200 | P99 latency SLA target |
| `lambda_sla` | 10.0 | SLA violation penalty weight |
| `lambda_variance` | 2.0 | Latency variance penalty weight |
| `volatility_alpha` | 1.5 | TR sensitivity to workload burstiness |
| `n_initial` | 20 | Sobol quasi-random initial samples |
| `n_iterations` | 50 | BO iterations after initialization |

---

## 🧮 Complexity

| Component | Time | Space |
|-----------|------|-------|
| Simulation | O(N × L) | O(N) |
| GP Fitting | O(n³) | O(n²) |
| Sensitivity | O(S × P × sim) | O(S × P) |
| Full Loop | O(I × (sim + n³)) | O(I × d) |

---

## 🏃 Tests

```bash
pytest tests/ -v
```

---

## ⚠️ Known Limitations

- **Simulation gap:** SimPy models may not reflect real cloud behavior
- **GP scalability:** Practical limit ~1000 observations
- **Extreme workloads:** Embedding may fail under chaotic patterns
- **Structural mislearning:** Incorrect clustering → suboptimal configs

---

## 📝 Tech Stack

- **Python 3.10+** — Core language
- **BoTorch** — GP surrogate + TuRBO foundation
- **SimPy** — Discrete event simulation
- **scikit-learn** — Spectral clustering
- **Chart.js** — Dashboard visualization

---

## 📄 License

MIT License

---

*InfraMIND v3 — Research-grade infrastructure intelligence that learns workload-aware optimization policies through structure discovery and adaptive Bayesian search.*
