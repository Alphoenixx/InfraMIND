from typing import List, Dict, Any
import time
from api.models.schemas import CodeCandidate, EvaluationScores, TrainingConfig
from api.services.evaluator_service import EvaluatorService
from api.services.llm_provider import LLMProviderService

class GeneratorService:
    """Generates infrastructure code using multiple LLMs."""
    
    def __init__(self):
        self.evaluator = EvaluatorService()
        self.default_weights = TrainingConfig()
        self.llm_provider = LLMProviderService()

    async def generate_async(self, prompt: str, target: str) -> List[CodeCandidate]:
        results = await self.llm_provider.generate_all(prompt, target)
        
        candidates = []
        for i, res in enumerate(results):
            code = res["code"]
            config = res["config"]
            provider = res["provider"]
            
            raw_scores = self.evaluator.evaluate(code, config, self.default_weights)
            scores = EvaluationScores(**raw_scores)
            
            # Clean ID spacing for UI mapping
            safe_id = provider.lower().replace(" ", "_").replace("(", "").replace(")", "")
            
            candidates.append(CodeCandidate(
                id=f"cand_{safe_id}_{int(time.time()*1000)}",
                provider=provider,
                code=code,
                language="hcl" if target == "terraform" else target,
                scores=scores
            ))
            
        return sorted(candidates, key=lambda c: c.scores.composite_score, reverse=True)
