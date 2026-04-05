from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Finding(BaseModel):
    id: str
    type: str  # "pass" | "warn" | "fail"
    category: str
    msg: str
    reason: str
    impact: str
    fix: Optional[str] = None

class EvaluationScores(BaseModel):
    syntax_score: float
    execution_result: str  # "success" | "error" | "warning"
    cost_estimate_usd: float
    security_score: float
    correctness_score: float
    composite_score: float
    findings: Optional[List[Finding]] = []
    deploy_readiness: Optional[str] = "yellow"
    baseline_delta: Optional[float] = 0.0
    confidence: Optional[str] = "medium"

class CodeCandidate(BaseModel):
    id: str
    provider: Optional[str] = None
    code: str
    language: str
    scores: EvaluationScores

class GenerateRequest(BaseModel):
    prompt: str
    target: str = "terraform"

class GenerateResponse(BaseModel):
    candidates: List[CodeCandidate]
    generation_time_ms: float

class SimulateRequest(BaseModel):
    config: Dict[str, Any]
    workload_type: str = "bursty"
    duration_s: float = 300.0

class TrainingConfig(BaseModel):
    alpha: float = 0.3
    beta: float = 0.25
    gamma: float = 0.2
    delta: float = 0.15
    epsilon: float = 0.1
    epochs: int = 50
    learning_rate: float = 0.001

class TrainingProgress(BaseModel):
    epoch: int
    total_epochs: int
    reward: float
    loss: float
    best_reward: float
    reward_history: List[float]
    status: str  # "running" | "stopped" | "completed"

class CompareRequest(BaseModel):
    baseline_config: Dict[str, Any]
    optimized_config: Dict[str, Any]
    workload_type: str = "bursty"
    duration_s: float = 300.0
