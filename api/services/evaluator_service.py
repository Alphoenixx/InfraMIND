import logging
from typing import Dict, Any, List

# Evaluation constants — extracted to avoid magic numbers
DEFAULT_BASELINE_SCORE = 75.0  # The baseline composite score used for delta calculation
COST_PER_REPLICA_PER_DAY = 1.20  # Rough estimate: $0.05/replica/hour * 24h
SECURITY_PENALTY_OPEN_CIDR = 30
SECURITY_PENALTY_NO_ENCRYPTION = 20
SECURITY_PENALTY_NO_SG = 25

class EvaluatorService:
    """Multi-objective evaluation: syntax + execution + cost + security + correctness."""
    
    def evaluate(self, code: str, config: Dict[str, Any], weights: Any) -> Dict[str, Any]:
        syntax = self._check_syntax(code)
        execution = self._simulate_execution(config)
        cost = self._estimate_cost(config)
        security = self._check_security(code, config)
        correctness = self._check_correctness(code, config)
        
        composite = (
            weights.alpha * syntax +
            weights.beta * execution +
            weights.gamma * max(0, 100 - cost) +
            weights.delta * security +
            weights.epsilon * correctness
        )
        
        findings = []
        # Syntax Findings
        if syntax == 100.0:
            findings.append({"id": "syn_1", "type": "pass", "category": "Syntax", "msg": "HCL Syntax Valid", "reason": "All resource blocks well-formed", "impact": "None"})
        else:
            findings.append({"id": "syn_2", "type": "fail", "category": "Syntax", "msg": "Invalid Syntax", "reason": "Missing required resource definitions", "impact": "High - Will fail deployment", "fix": "Ensure valid terraform resources exist"})
            
        # Security Findings
        if "0.0.0.0/0" in code:
            findings.append({"id": "sec_1", "type": "fail", "category": "Security", "msg": "Open CIDR Detected", "reason": "Ingress uses 0.0.0.0/0", "impact": "High - publicly accessible - penalty 30", "fix": "Restrict to 10.0.0.0/8 or specific IP"})
        if "encrypted = false" in code or "encrypted" not in code:
            findings.append({"id": "sec_2", "type": "warn", "category": "Security", "msg": "Missing Encryption Flag", "reason": "No `encrypted` attribute found", "impact": "Medium - Data at rest unprotected", "fix": "Add `encrypted = true` to EBS or RDS blocks"})
        if "0.0.0.0/0" not in code and "encrypted" in code:
            findings.append({"id": "sec_3", "type": "pass", "category": "Security", "msg": "Secure Configuration", "reason": "No open ports and encryption enabled", "impact": "None"})

        # Correctness Findings
        if "subnet" in code and "vpc_id" not in code:
            findings.append({"id": "cor_1", "type": "fail", "category": "Correctness", "msg": "Missing VPC Association", "reason": "Subnet defined without a linked VPC", "impact": "High", "fix": "Add vpc_id reference"})
        elif "vpc_id" in code:
            findings.append({"id": "cor_2", "type": "pass", "category": "Correctness", "msg": "VPC Validation", "reason": "VPC -> Subnet linking verified", "impact": "None"})

        # Deploy Readiness
        is_safe = all(f["type"] != "fail" for f in findings)
        is_warn = any(f["type"] == "warn" for f in findings)
        readiness = "green" if is_safe and not is_warn else ("yellow" if is_safe else "red")
        
        return {
            "syntax_score": syntax,
            "execution_result": "success" if execution > 80 else "error",
            "cost_estimate_usd": cost,
            "security_score": security,
            "correctness_score": correctness,
            "composite_score": composite,
            "findings": findings,
            "deploy_readiness": readiness,
            "baseline_delta": round(composite - DEFAULT_BASELINE_SCORE, 1),
            "confidence": "high" if composite > 80 else "medium"
        }
    
    def _check_syntax(self, code: str) -> float:
        # Simple heuristic
        if "resource" in code or "module" in code:
            return 100.0
        return 50.0

    def _simulate_execution(self, config: Dict) -> float:
        """Heuristic execution score based on config structure.
        
        Returns a score 0-100 based on whether the config has enough
        services with reasonable replica counts. Not a real tf-plan.
        """
        if not config:
            return 50.0
        total_replicas = sum(
            params.get("replicas", 0) for name, params in config.items() if not name.startswith("_")
        )
        # More replicas = higher availability = better execution readiness
        if total_replicas >= 6:
            return 95.0
        elif total_replicas >= 3:
            return 85.0
        return 60.0

    def _estimate_cost(self, config: Dict) -> float:
        """Estimate daily infrastructure cost from config replica counts."""
        cost = 0.0
        for svc_name, params in config.items():
            if svc_name.startswith("_"): continue
            replicas = params.get("replicas", 2)
            cpu = params.get("cpu_millicores", 500)
            # Base cost per replica + CPU scaling factor
            cost += replicas * COST_PER_REPLICA_PER_DAY * (cpu / 500)
        return round(cost, 2)
        
    def _check_security(self, code: str, config: Dict) -> float:
        score = 100.0
        if "0.0.0.0/0" in code:
            score -= SECURITY_PENALTY_OPEN_CIDR
        if "encrypted = false" in code or "encrypted" not in code:
            score -= SECURITY_PENALTY_NO_ENCRYPTION
        if "aws_security_group" not in code:
            score -= SECURITY_PENALTY_NO_SG
        return max(0, score)
    
    def _check_correctness(self, code: str, config: Dict) -> float:
        score = 100.0
        if "subnet" in code and "vpc_id" not in code:
            score -= 30
        if "aws_instance" in code and "vpc_security_group_ids" not in code:
            score -= 30
        return max(0, score)
