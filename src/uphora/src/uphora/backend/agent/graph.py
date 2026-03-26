from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    make_memory_loader_node,
    make_intent_router_node,
    make_advisor_node,
    make_shopper_node,
    make_coach_node,
)


def _route_intent(state: AgentState) -> str:
    return state["intent"] or "advisor"


def build_graph(llm):
    """Build and compile the LangGraph agent graph with injected LLM."""
    g = StateGraph(AgentState)

    g.add_node("memory_loader", make_memory_loader_node())
    g.add_node("intent_router", make_intent_router_node())
    g.add_node("advisor",       make_advisor_node(llm))
    g.add_node("shopper",       make_shopper_node(llm))
    g.add_node("coach",         make_coach_node(llm))

    g.set_entry_point("memory_loader")
    g.add_edge("memory_loader", "intent_router")
    g.add_conditional_edges(
        "intent_router",
        _route_intent,
        {"advisor": "advisor", "shopper": "shopper", "coach": "coach"},
    )
    g.add_edge("advisor", END)
    g.add_edge("shopper", END)
    g.add_edge("coach",   END)

    return g.compile()
