from app.modules.graph.states.chat_state import ChatState

def ticket_node(state: ChatState) -> ChatState:
    state["answer"] = "CEK TIKET BERHASIL"
    return state