from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import generation, simulation, dag, comparison
from api.websockets import training_ws

app = FastAPI(title="InfraMIND v4 API", version="4.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generation.router, prefix="/api")
app.include_router(simulation.router, prefix="/api")
app.include_router(dag.router, prefix="/api")
app.include_router(training_ws.router, prefix="/api")
app.include_router(comparison.router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
