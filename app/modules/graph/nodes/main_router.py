from pydantic import BaseModel, Field
from typing import Literal
from app.modules.graph.states.chat_state import ChatState
from app.services.llm import get_llm
from app.services.prompt import prompt_template

class RouterOutput(BaseModel):
    category: Literal["info", "lapor", "ticket"] = Field(description="intent classification results")

llm = get_llm(max_tokens=50)
router_llm = llm.with_structured_output(RouterOutput)
router_prompt = prompt_template.get_prompt("router")
router_pipeline = router_prompt | router_llm

def main_router_node(state: ChatState) -> RouterOutput:
    router_output = router_pipeline.invoke({"question": state["question"]})
    state["category"] = router_output.category
    return state

def main_router_decision(state: ChatState) -> str:
    if state["category"] == "info":
        return "rag_edge"
    elif state["category"] == "lapor":
        return "lapor_edge"
    elif state["category"] == "ticket":
        return "ticket_edge"
    else:
        raise ValueError("Kategori query tidak valid")