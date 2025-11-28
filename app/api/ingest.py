from fastapi import APIRouter, BackgroundTasks
from app.schemas.ingest_schema import IngestRequest
from app.modules.ingestion.ingestion_pipeline import run_ingestion_pipeline

ingest_router = APIRouter()

@ingest_router.post("/ingest")
async def ingest(request: IngestRequest, background: BackgroundTasks):
    background.add_task(run_ingestion_pipeline, request.folder_path)
    return {"status": "queued",
            "message": "Ingestion started in background",
            "path": request.folder_path
            }