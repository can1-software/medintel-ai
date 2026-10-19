from fastapi import FastAPI

app = FastAPI(
    title="MedIntel AI",
    version="0.1.0",
    description="Medical Document Intelligence Platform"
)

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}