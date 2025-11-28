from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.modules.rag.rag_pipeline import RAGPipeline
from app.schemas.chat_schema import ChatRequest
from app.core.config import settings

chat_router = APIRouter()
rag = RAGPipeline(settings.PROMPT_VERSION)

@chat_router.post("/chat")
async def chat(request: ChatRequest):
    pipeline = rag.chat()
    async def stream_answer():
        async for chunk in pipeline.astream({"question": request.question}):
            yield chunk
    return StreamingResponse(stream_answer(), media_type="text/plain")