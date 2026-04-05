from fastapi import APIRouter, HTTPException
import time
from api.models.schemas import GenerateRequest, GenerateResponse
from api.services.generator_service import GeneratorService

router = APIRouter(prefix="/generate", tags=["generation"])
generator_service = GeneratorService()

@router.post("", response_model=GenerateResponse)
async def generate_code(request: GenerateRequest):
    start_time = time.time()
    
    try:
        candidates = await generator_service.generate_async(
            prompt=request.prompt,
            target=request.target
        )
        
        elapsed = (time.time() - start_time) * 1000  # ms
        
        return GenerateResponse(
            candidates=candidates,
            generation_time_ms=elapsed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
