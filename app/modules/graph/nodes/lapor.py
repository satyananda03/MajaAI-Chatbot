from app.modules.graph.states.chat_state import ChatState

def lapor_node(state: ChatState) -> ChatState:
    state["answer"] = "LAPOR BERHASIL"
    return state