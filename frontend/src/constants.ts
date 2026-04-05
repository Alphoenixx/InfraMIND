/// <reference types="vite/client" />

// Shared constants and utilities for InfraMIND frontend

// API base URL — change this to your deployed backend URL in production
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

// Weight slider definitions — shared between EvaluationPanel and TrainingDashboard
export const WEIGHT_SLIDERS = [
    { key: 'alpha', label: 'Syntax', color: 'bg-accent-green' },
    { key: 'beta', label: 'Execution', color: 'bg-accent-blue' },
    { key: 'gamma', label: 'Cost', color: 'bg-accent-orange' },
    { key: 'delta', label: 'Security', color: 'bg-accent-purple' },
    { key: 'epsilon', label: 'Correctness', color: 'bg-accent-cyan' },
] as const;

// Default weight values
export const DEFAULT_WEIGHTS = {
    alpha: 0.3,
    beta: 0.25,
    gamma: 0.2,
    delta: 0.15,
    epsilon: 0.1,
};

// Compute composite score from raw scores and weights
export function computeComposite(
    scores: { syntax_score: number; execution_result: string; cost_estimate_usd: number; security_score: number; correctness_score: number },
    weights: { alpha: number; beta: number; gamma: number; delta: number; epsilon: number }
): number {
    return (
        weights.alpha * scores.syntax_score +
        weights.beta * (scores.execution_result === 'success' ? 100 : 0) +
        weights.gamma * Math.max(0, 100 - scores.cost_estimate_usd) +
        weights.delta * scores.security_score +
        weights.epsilon * scores.correctness_score
    );
}
