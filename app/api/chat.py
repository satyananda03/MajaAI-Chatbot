from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat_schema import ChatRequest
from app.modules.graph.workflow import workflow

chat_router = APIRouter()

@chat_router.post("/chat")
async def chat(request: ChatRequest):
    async def stream_answer():
        inputs = {"question": request.question}
        async for event in workflow.astream_events(inputs, version="v2"):
            kind = event["event"]
            # Streaming token dari LLM di RAG Node
            if kind == "on_chat_model_stream":
                node_name = event["metadata"].get("langgraph_node") 
                if node_name == "rag_node":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                        if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                            yield content[0]["text"]
            # Output dari lapor & ticket node
            elif kind == "on_chain_end":
                node_name = event["metadata"].get("langgraph_node")
                if node_name in ["lapor_node", "ticket_node"]:
                    output = event["data"].get("output")
                    if output and 'answer' in output:
                        yield output['answer'] 
    return StreamingResponse(stream_answer(), media_type="text/plain",headers={
                                                                                "Cache-Control": "no-cache",
                                                                                "Connection": "keep-alive",
                                                                                "X-Accel-Buffering": "no",
                                                                                })