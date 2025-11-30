from langgraph.graph import StateGraph, START, END
from app.modules.graph.states.chat_state import ChatState
from app.modules.graph.nodes.main_router import main_router_node, main_router_decision
from app.modules.graph.nodes.rag import rag_node
from app.modules.graph.nodes.lapor import lapor_node
from app.modules.graph.nodes.ticket import ticket_node

graph = StateGraph(ChatState)
graph.add_node("main_router_node", main_router_node)
graph.add_node("rag_node", rag_node)
graph.add_node("lapor_node", lapor_node) 
graph.add_node("ticket_node", ticket_node)

graph.add_edge(START, "main_router_node")
graph.add_conditional_edges(
    "main_router_node",
    main_router_decision,
    {"rag_edge": "rag_node", 
    "lapor_edge": "lapor_node", 
    "ticket_edge": "ticket_node"}
)
graph.add_edge("rag_node", END)
graph.add_edge("lapor_node", END)
graph.add_edge("ticket_node", END)

workflow = graph.compile()