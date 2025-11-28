from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import chat_router
from app.api.ingest import ingest_router
from app.api.evaluate import evaluate_router
from app.core.logging import setup_logging

setup_logging()
app = FastAPI()

app.add_middleware(CORSMiddleware,
                    allow_origins=["*"],              
                    allow_credentials=True,            
                    allow_methods=["GET", "POST"],          
                    allow_headers=["*"])
                    
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(evaluate_router)
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000