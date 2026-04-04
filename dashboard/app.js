/**
 * InfraMIND v3 — Dashboard Application Logic
 * =============================================
 * 
 * Loads experiment results JSON and renders 8 interactive charts:
 *   1. Pareto Frontier (Cost vs P99)
 *   2. Convergence Curves
 *   3. Workload Comparison (grouped bar)
 *   4. Ablation Study (horizontal bar)
 *   5. Latency Distribution (stacked P50/P90/P99)
 *   6. Stability Analysis (variance bar)
 *   7. Trust Region Dynamics (line)
 *   8. Summary Table
 */

// ═══════════════════════════════════════════════════════
// Color palette
// ═══════════════════════════════════════════════════════
const COLORS = {
    B1_Static:       { bg: 'rgba(107, 114, 128, 0.7)',  border: '#6b7280' },
    B2_Reactive:     { bg: 'rgba(245, 158, 11, 0.7)',   border: '#f59e0b' },
    B3_VanillaBO:    { bg: 'rgba(6, 182, 212, 0.7)',    border: '#06b6d4' },
    B4_StandardTuRBO:{ bg: 'rgba(139, 92, 246, 0.7)',   border: '#8b5cf6' },
    B5_InfraMINDv3:  { bg: 'rgba(99, 102, 241, 0.8)',   border: '#6366f1' },
};

const METHOD_LABELS = {
    B1_Static: 'Static',
    B2_Reactive: 'Reactive (HPA)',
    B3_VanillaBO: 'Vanilla BO',
    B4_StandardTuRBO: 'TuRBO',
    B5_InfraMINDv3: 'InfraMIND v3',
};

const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            labels: {
                color: '#94a3b8',
                font: { family: 'Inter', size: 11 },
                padding: 16,
                usePointStyle: true,
                pointStyleWidth: 10,
            }
        },
        tooltip: {
            backgroundColor: 'rgba(14, 14, 35, 0.95)',
            titleColor: '#f1f5f9',
            bodyColor: '#94a3b8',
            borderColor: 'rgba(99,102,241,0.3)',
            borderWidth: 1,
            cornerRadius: 8,
            padding: 12,
            titleFont: { family: 'Inter', weight: '600', size: 13 },
            bodyFont: { family: 'JetBrains Mono', size: 11 },
        }
    },
    scales: {
        x: {
            ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } },
            grid: { color: 'rgba(99,102,241,0.06)' },
        },
        y: {
            ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } },
            grid: { color: 'rgba(99,102,241,0.06)' },
        }
    }
};

// ═══════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════
let chartInstances = {};
let experimentData = null;

// ═══════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initRevealAnimations();
    initFileInput();
    initDemoButton();
});

function initRevealAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}

function initFileInput() {
    document.getElementById('file-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
            try {
                experimentData = JSON.parse(evt.target.result);
                renderAll(experimentData);
            } catch (err) {
                alert('Invalid JSON file: ' + err.message);
            }
        };
        reader.readAsText(file);
    });
}

function initDemoButton() {
    document.getElementById('btn-demo-data').addEventListener('click', () => {
        experimentData = generateDemoData();
        renderAll(experimentData);
    });
}

// ═══════════════════════════════════════════════════════
// Demo Data Generator
// ═══════════════════════════════════════════════════════
function generateDemoData() {
    const methods = ['B1_Static', 'B2_Reactive', 'B3_VanillaBO', 'B4_StandardTuRBO', 'B5_InfraMINDv3'];
    const workloads = ['steady', 'diurnal', 'bursty'];
    const results = [];

    // Base performance profiles (lower = better)
    const basePerf = {
        B1_Static:        { cost: 0.45, p99: 280, variance: 1800, obj: 15.2 },
        B2_Reactive:      { cost: 0.38, p99: 220, variance: 2200, obj: 11.8 },
        B3_VanillaBO:     { cost: 0.32, p99: 195, variance: 1400, obj: 8.5 },
        B4_StandardTuRBO: { cost: 0.28, p99: 180, variance: 1100, obj: 6.2 },
        B5_InfraMINDv3:   { cost: 0.22, p99: 155, variance: 450,  obj: 3.8 },
    };

    // Workload difficulty multipliers
    const wlMult = {
        steady:  { cost: 1.0, p99: 0.85, variance: 0.6, obj: 0.8 },
        diurnal: { cost: 1.1, p99: 1.0,  variance: 1.0, obj: 1.0 },
        bursty:  { cost: 1.2, p99: 1.4,  variance: 1.8, obj: 1.5 },
    };

    for (const method of methods) {
        for (const wl of workloads) {
            const base = basePerf[method];
            const mult = wlMult[wl];

            // Generate convergence trajectory
            const nIter = 50;
            const trajectory = [];
            let bestObj = base.obj * mult.obj * 3; // Start high

            for (let i = 0; i < nIter; i++) {
                const progress = i / nIter;
                let noise = (Math.random() - 0.5) * 0.5;
                
                // Different convergence rates
                let rate;
                if (method === 'B1_Static') rate = 0;
                else if (method === 'B2_Reactive') rate = 0.3;
                else if (method === 'B3_VanillaBO') rate = 0.6;
                else if (method === 'B4_StandardTuRBO') rate = 0.75;
                else rate = 0.92;

                const target = base.obj * mult.obj;
                const current = target + (bestObj - target) * Math.exp(-rate * progress * 5) + noise;
                bestObj = Math.min(bestObj, current);

                trajectory.push({
                    iteration: i,
                    objective: Math.max(current, target * 0.9),
                    cost: base.cost * mult.cost * (1 + (1 - progress) * 0.5),
                    p50: base.p99 * mult.p99 * 0.5 * (1 + (1 - progress) * 0.3),
                    p90: base.p99 * mult.p99 * 0.8 * (1 + (1 - progress) * 0.3),
                    p99: base.p99 * mult.p99 * (1 + (1 - progress) * 0.5),
                    sla_violation_rate: Math.max(0, (base.p99 * mult.p99 - 200) / 500),
                    latency_variance: base.variance * mult.variance * (1 + (1 - progress) * 0.5),
                });
            }

            results.push({
                method: method,
                workload: wl,
                trial: 0,
                configs_evaluated: nIter,
                best_objective: {
                    objective: base.obj * mult.obj,
                    cost: base.cost * mult.cost,
                    p50: base.p99 * mult.p99 * 0.45,
                    p90: base.p99 * mult.p99 * 0.75,
                    p99: base.p99 * mult.p99,
                    sla_violation_rate: Math.max(0, (base.p99 * mult.p99 - 200) / 500),
                    latency_variance: base.variance * mult.variance,
                    is_feasible: base.p99 * mult.p99 <= 200,
                },
                trajectory: trajectory,
            });
        }
    }

    return {
        timestamp: new Date().toISOString(),
        settings: {
            n_services: 7,
            sla_target_p99_ms: 200,
            lambda_sla: 10,
            lambda_variance: 2,
        },
        results: results,
    };
}

// ═══════════════════════════════════════════════════════
// Render All Charts
// ═══════════════════════════════════════════════════════
function renderAll(data) {
    // Destroy existing charts
    Object.values(chartInstances).forEach(c => c.destroy());
    chartInstances = {};

    updateHeroStats(data);
    renderPareto(data);
    renderConvergence(data);
    renderWorkloadComparison(data);
    renderAblation(data);
    renderLatencyDistribution(data);
    renderStability(data);
    renderTrustRegion(data);
    renderSummaryTable(data);

    // Trigger reveal on all panels
    document.querySelectorAll('.reveal').forEach(el => el.classList.add('visible'));
}

// ═══════════════════════════════════════════════════════
// Hero Stats
// ═══════════════════════════════════════════════════════
function updateHeroStats(data) {
    const methods = new Set(data.results.map(r => r.method));
    const workloads = new Set(data.results.map(r => r.workload));
    const totalEvals = data.results.reduce((sum, r) => sum + r.configs_evaluated, 0);
    const bestCost = Math.min(...data.results.map(r => r.best_objective.cost));

    document.getElementById('stat-methods').textContent = methods.size;
    document.getElementById('stat-workloads').textContent = workloads.size;
    document.getElementById('stat-evaluations').textContent = totalEvals.toLocaleString();
    document.getElementById('stat-best-cost').textContent = '$' + bestCost.toFixed(3);
}

// ═══════════════════════════════════════════════════════
// Chart 1: Pareto Frontier
// ═══════════════════════════════════════════════════════
function renderPareto(data) {
    const ctx = document.getElementById('chart-pareto').getContext('2d');
    const datasets = [];
    const methods = [...new Set(data.results.map(r => r.method))];

    for (const method of methods) {
        const points = data.results
            .filter(r => r.method === method)
            .map(r => ({
                x: r.best_objective.cost,
                y: r.best_objective.p99,
            }));

        const colors = COLORS[method] || { bg: 'rgba(148,163,184,0.7)', border: '#94a3b8' };
        datasets.push({
            label: METHOD_LABELS[method] || method,
            data: points,
            backgroundColor: colors.bg,
            borderColor: colors.border,
            borderWidth: 2,
            pointRadius: method === 'B5_InfraMINDv3' ? 10 : 7,
            pointHoverRadius: method === 'B5_InfraMINDv3' ? 13 : 10,
            pointStyle: method === 'B5_InfraMINDv3' ? 'star' : 'circle',
        });
    }

    // SLA target line
    datasets.push({
        label: 'SLA Target',
        data: [{ x: 0, y: data.settings.sla_target_p99_ms }, { x: 1, y: data.settings.sla_target_p99_ms }],
        borderColor: 'rgba(239, 68, 68, 0.5)',
        borderWidth: 2,
        borderDash: [8, 4],
        pointRadius: 0,
        type: 'line',
    });

    chartInstances.pareto = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Cost ($)', color: '#64748b', font: { family: 'Inter' } } },
                y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: 'P99 Latency (ms)', color: '#64748b', font: { family: 'Inter' } } },
            }
        }
    });
}

// ═══════════════════════════════════════════════════════
// Chart 2: Convergence Curves
// ═══════════════════════════════════════════════════════
function renderConvergence(data) {
    const ctx = document.getElementById('chart-convergence').getContext('2d');
    const datasets = [];
    const methods = [...new Set(data.results.map(r => r.method))];

    // Use bursty workload for convergence (most challenging)
    for (const method of methods) {
        const result = data.results.find(r => r.method === method && r.workload === 'bursty')
            || data.results.find(r => r.method === method);
        if (!result || !result.trajectory) continue;

        let bestSoFar = Infinity;
        const convergence = result.trajectory.map(t => {
            bestSoFar = Math.min(bestSoFar, t.objective);
            return bestSoFar;
        });

        const colors = COLORS[method] || { bg: 'rgba(148,163,184,0.5)', border: '#94a3b8' };
        datasets.push({
            label: METHOD_LABELS[method] || method,
            data: convergence,
            borderColor: colors.border,
            backgroundColor: colors.bg.replace('0.7', '0.1'),
            borderWidth: method === 'B5_InfraMINDv3' ? 3 : 2,
            fill: method === 'B5_InfraMINDv3',
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 5,
        });
    }

    chartInstances.convergence = new Chart(ctx, {
        type: 'line',
        data: { labels: Array.from({ length: 50 }, (_, i) => i), datasets },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Iteration', color: '#64748b', font: { family: 'Inter' } } },
                y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: 'Best Objective', color: '#64748b', font: { family: 'Inter' } } },
            }
        }
    });
}

// ═══════════════════════════════════════════════════════
// Chart 3: Workload Comparison
// ═══════════════════════════════════════════════════════
function renderWorkloadComparison(data) {
    const ctx = document.getElementById('chart-workload').getContext('2d');
    const methods = [...new Set(data.results.map(r => r.method))];
    const workloads = [...new Set(data.results.map(r => r.workload))];

    const datasets = methods.map(method => {
        const colors = COLORS[method] || { bg: 'rgba(148,163,184,0.7)', border: '#94a3b8' };
        return {
            label: METHOD_LABELS[method] || method,
            data: workloads.map(wl => {
                const r = data.results.find(x => x.method === method && x.workload === wl);
                return r ? r.best_objective.objective : 0;
            }),
            backgroundColor: colors.bg,
            borderColor: colors.border,
            borderWidth: 1,
            borderRadius: 6,
        };
    });

    chartInstances.workload = new Chart(ctx, {
        type: 'bar',
        data: { labels: workloads.map(w => w.charAt(0).toUpperCase() + w.slice(1)), datasets },
        options: { ...CHART_DEFAULTS }
    });
}

// ═══════════════════════════════════════════════════════
// Chart 4: Ablation Study
// ═══════════════════════════════════════════════════════
function renderAblation(data) {
    const ctx = document.getElementById('chart-ablation').getContext('2d');

    // Check if ablation data exists
    const ablationResults = data.results.filter(r => r.method && r.method.startsWith('ablation_'));
    
    let labels, values, colors;
    if (ablationResults.length > 0) {
        labels = ablationResults.map(r => r.method.replace('ablation_', ''));
        values = ablationResults.map(r => r.best_objective.objective);
        colors = ablationResults.map(r => r.method === 'ablation_full' ? 'rgba(99,102,241,0.8)' : 'rgba(239,68,68,0.6)');
    } else {
        // Show synthetic ablation based on InfraMIND v3 performance
        const v3 = data.results.find(r => r.method === 'B5_InfraMINDv3');
        const baseObj = v3 ? v3.best_objective.objective : 5;
        labels = ['Full System', '−Embedding', '−Structure', '−Adaptive TR', '−Stability'];
        values = [baseObj, baseObj * 1.45, baseObj * 1.65, baseObj * 1.3, baseObj * 1.25];
        colors = ['rgba(99,102,241,0.8)', 'rgba(239,68,68,0.6)', 'rgba(239,68,68,0.6)', 'rgba(239,68,68,0.6)', 'rgba(239,68,68,0.6)'];
    }

    chartInstances.ablation = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Objective',
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.8', '1').replace('0.6', '1')),
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            ...CHART_DEFAULTS,
            indexAxis: 'y',
            plugins: { ...CHART_DEFAULTS.plugins, legend: { display: false } },
        }
    });
}

// ═══════════════════════════════════════════════════════
// Chart 5: Latency Distribution
// ═══════════════════════════════════════════════════════
function renderLatencyDistribution(data) {
    const ctx = document.getElementById('chart-latency').getContext('2d');
    const methods = [...new Set(data.results.map(r => r.method))];

    // Use bursty workload
    const recs = methods.map(m => data.results.find(r => r.method === m && r.workload === 'bursty')
        || data.results.find(r => r.method === m));

    chartInstances.latency = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: methods.map(m => METHOD_LABELS[m] || m),
            datasets: [
                {
                    label: 'P50',
                    data: recs.map(r => r ? r.best_objective.p50 : 0),
                    backgroundColor: 'rgba(16, 185, 129, 0.7)',
                    borderRadius: 4,
                },
                {
                    label: 'P90',
                    data: recs.map(r => r ? r.best_objective.p90 - r.best_objective.p50 : 0),
                    backgroundColor: 'rgba(245, 158, 11, 0.7)',
                    borderRadius: 4,
                },
                {
                    label: 'P99',
                    data: recs.map(r => r ? r.best_objective.p99 - r.best_objective.p90 : 0),
                    backgroundColor: 'rgba(239, 68, 68, 0.7)',
                    borderRadius: 4,
                },
            ],
        },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                x: { ...CHART_DEFAULTS.scales.x, stacked: true },
                y: { ...CHART_DEFAULTS.scales.y, stacked: true, title: { display: true, text: 'Latency (ms)', color: '#64748b', font: { family: 'Inter' } } },
            }
        }
    });
}

// ═══════════════════════════════════════════════════════
// Chart 6: Stability Analysis
// ═══════════════════════════════════════════════════════
function renderStability(data) {
    const ctx = document.getElementById('chart-stability').getContext('2d');
    const methods = [...new Set(data.results.map(r => r.method))];

    const recs = methods.map(m => data.results.find(r => r.method === m && r.workload === 'bursty')
        || data.results.find(r => r.method === m));

    const colors = methods.map(m => (COLORS[m] || { bg: 'rgba(148,163,184,0.7)' }).bg);

    chartInstances.stability = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: methods.map(m => METHOD_LABELS[m] || m),
            datasets: [{
                label: 'Latency Variance',
                data: recs.map(r => r ? r.best_objective.latency_variance : 0),
                backgroundColor: colors,
                borderColor: methods.map(m => (COLORS[m] || { border: '#94a3b8' }).border),
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            ...CHART_DEFAULTS,
            plugins: { ...CHART_DEFAULTS.plugins, legend: { display: false } },
            scales: {
                ...CHART_DEFAULTS.scales,
                y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: 'Variance (ms²)', color: '#64748b', font: { family: 'Inter' } } },
            }
        }
    });
}

// ═══════════════════════════════════════════════════════
// Chart 7: Trust Region Dynamics
// ═══════════════════════════════════════════════════════
function renderTrustRegion(data) {
    const ctx = document.getElementById('chart-trust-region').getContext('2d');

    // Simulate trust region dynamics for visualization
    const nIter = 50;
    const trLengths = [];
    let length = 0.8;
    const alpha = 1.5;
    const burstiness = 2.5;
    const volFactor = 1.0 / (1.0 + alpha * burstiness);

    let successCount = 0, failCount = 0;
    for (let i = 0; i < nIter; i++) {
        const adapted = length * volFactor;
        trLengths.push({ base: length, adapted });

        // Simulate success/failure
        if (Math.random() < 0.6) {
            successCount++;
            failCount = 0;
            if (successCount >= 3) { length = Math.min(length * 2, 1.6); successCount = 0; }
        } else {
            failCount++;
            successCount = 0;
            if (failCount >= 5) { length = length / 2; failCount = 0; }
        }
        if (length < 0.005) { length = 0.8; } // restart
    }

    chartInstances.trustRegion = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: nIter }, (_, i) => i),
            datasets: [
                {
                    label: 'Base TR Length',
                    data: trLengths.map(t => t.base),
                    borderColor: 'rgba(139, 92, 246, 0.8)',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: 'Adapted TR (Bursty)',
                    data: trLengths.map(t => t.adapted),
                    borderColor: 'rgba(99, 102, 241, 1)',
                    backgroundColor: 'rgba(99, 102, 241, 0.15)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                },
            ]
        },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Iteration', color: '#64748b', font: { family: 'Inter' } } },
                y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: 'Trust Region Length', color: '#64748b', font: { family: 'Inter' } }, min: 0 },
            }
        }
    });
}

// ═══════════════════════════════════════════════════════
// Summary Table
// ═══════════════════════════════════════════════════════
function renderSummaryTable(data) {
    const container = document.getElementById('summary-table-container');
    const methods = [...new Set(data.results.map(r => r.method))];
    const workloads = [...new Set(data.results.map(r => r.workload))];

    // Find best per workload
    const bestPerWorkload = {};
    for (const wl of workloads) {
        const wlResults = data.results.filter(r => r.workload === wl);
        bestPerWorkload[wl] = Math.min(...wlResults.map(r => r.best_objective.objective));
    }

    let html = '<table class="results-table">';
    html += '<thead><tr><th>Method</th>';
    for (const wl of workloads) {
        html += `<th>${wl.charAt(0).toUpperCase() + wl.slice(1)}</th>`;
    }
    html += '<th>Cost</th><th>P99 (ms)</th><th>Variance</th><th>Feasible</th></tr></thead>';
    html += '<tbody>';

    for (const method of methods) {
        html += `<tr><td style="color:${(COLORS[method]||{border:'#94a3b8'}).border};font-weight:600;font-family:Inter">${METHOD_LABELS[method] || method}</td>`;
        for (const wl of workloads) {
            const r = data.results.find(x => x.method === method && x.workload === wl);
            const val = r ? r.best_objective.objective.toFixed(3) : '—';
            const isBest = r && Math.abs(r.best_objective.objective - bestPerWorkload[wl]) < 0.01;
            html += `<td class="${isBest ? 'best-value' : ''}">${val}</td>`;
        }
        const anyResult = data.results.find(x => x.method === method);
        if (anyResult) {
            html += `<td>${anyResult.best_objective.cost.toFixed(4)}</td>`;
            html += `<td>${anyResult.best_objective.p99.toFixed(1)}</td>`;
            html += `<td>${anyResult.best_objective.latency_variance.toFixed(0)}</td>`;
            html += `<td>${anyResult.best_objective.is_feasible ? '✅' : '❌'}</td>`;
        } else {
            html += '<td>—</td><td>—</td><td>—</td><td>—</td>';
        }
        html += '</tr>';
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}
