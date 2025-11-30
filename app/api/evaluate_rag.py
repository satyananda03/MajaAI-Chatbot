from fastapi import APIRouter
from app.modules.rag.rag_pipeline import RAGPipeline
from app.schemas.evaluate_schema import EvaluateRequest, EvaluateResponse
import time

evaluate_rag_router = APIRouter()
rag = RAGPipeline()

# @evaluate_router.post("/evaluate_rag")
# async def evaluate(request: EvaluateRequest) -> EvaluateResponse: 
#     pipeline = rag.evaluate(settings.PROMPT_VERSION)
#     result = pipeline.invoke({"question": request.question})
#     context_list = result["context"] if isinstance(result["context"], list) else [result["context"]]
#     return EvaluateResponse(
#                             answer=result["answer"], 
#                             context=context_list
#                             )

@evaluate_rag_router.post("/evaluate_retriever")
async def evaluate_retriever(request: EvaluateRequest) -> EvaluateResponse: 
    t1 = time.time()
    pipeline = rag.evaluate_retriever()
    result = pipeline.invoke({"question": request.question})
    context_list = result["context"] if isinstance(result["context"], list) else [result["context"]]
    latency = time.time() - t1
    return EvaluateResponse(
                            context=context_list,
                            latency=latency
                            )