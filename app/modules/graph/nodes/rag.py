from app.modules.rag.rag_pipeline import RAGPipeline
from app.modules.graph.states.chat_state import ChatState

rag = RAGPipeline()
# rag_node.py
async def rag_node(state: ChatState) -> ChatState:
    output = await rag.chat().ainvoke({"question": state["question"]})
    state["answer"] = output
    return state