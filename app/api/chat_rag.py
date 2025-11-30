from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.modules.rag.rag_pipeline import RAGPipeline
from app.schemas.chat_schema import ChatRequest

chat_rag_router = APIRouter()
rag = RAGPipeline()

@chat_rag_router.post("/chat_rag")
async def chat_rag(request: ChatRequest):
    pipeline = rag.chat()
    async def stream_answer():
        async for chunk in pipeline.astream({"question": request.question}):
            yield chunk
    return StreamingResponse(stream_answer(), media_type="text/plain")