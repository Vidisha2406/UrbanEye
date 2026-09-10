import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure root directory and backend directory are on python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(BACKEND_DIR)

for path in [ROOT_DIR, BACKEND_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Load environment variables from root or backend directory if present
load_dotenv(os.path.join(ROOT_DIR, ".env"))
load_dotenv(os.path.join(BACKEND_DIR, ".env"))

from backend.api.upload import router as upload_router

app = FastAPI(
    title="UrbanEye Edge AI Inference Service",
    description="Real-time multi-model perception, kinematics, and telemetry pipeline for UrbanEye smart city transit fleet.",
    version="1.0.0",
)

# Safe CORS configuration for production and local development
cors_origins_env = os.getenv("ALLOWED_ORIGINS", os.getenv("CORS_ORIGINS", "")).strip()
if cors_origins_env:
    allowed_origins = [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

# If wildcard is explicitly specified, disable credentials to adhere to standard browser security specs
allow_all_origins = "*" in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else allowed_origins,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

EVIDENCE_DIR = os.path.join(BACKEND_DIR, "static", "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Mount static directory for annotated evidence frames
app.mount("/api/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence")
app.mount("/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence_direct")

# Include API routes
app.include_router(upload_router)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "UrbanEye Edge AI Backend",
        "docs": "/docs",
        "health": "/api/health",
        "upload_endpoint": "/api/upload",
        "detect_frame_endpoint": "/api/detect-frame",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "UrbanEye Edge AI Backend",
        "api_health": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port, reload=True)
