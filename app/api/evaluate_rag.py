from fastapi import APIRouter
from app.modules.rag.rag_pipeline import RAGPipeline
from app.schemas.evaluate_schema import EvaluateRequest, EvaluateResponse
import time

evaluate_rag_router = APIRouter()
rag = RAGPipeline()

@evaluate_rag_router.post("/evaluate_generator")
async def evaluate(request: EvaluateRequest) -> EvaluateResponse: 
    pipeline = rag.evaluate() 
    t1 = time.time()
    result = await pipeline.ainvoke({"question": request.question})
    latency = time.time() - t1
    context_list = result["context"] if isinstance(result["context"], list) else [result["context"]]
    return EvaluateResponse(answer=result["answer"], context=context_list, latency=latency)

@evaluate_rag_router.post("/evaluate_retriever")
async def evaluate_retriever(request: EvaluateRequest) -> EvaluateResponse: 
    pipeline = rag.evaluate_retriever()
    t1 = time.time()
    result = await pipeline.ainvoke({"question": request.question})
    latency = time.time() - t1
    context_list = result["context"] if isinstance(result["context"], list) else [result["context"]]
    return EvaluateResponse(answer=None, context=context_list, latency=latency)