"""
LangGraph node factory functions for the Uphora beauty agent.
All nodes are synchronous. Each specialist node calls the LLM directly via ChatDatabricks.
Nodes are created via make_*_node(llm) closures so the LLM is injected by build_graph().
"""
from langchain_core.messages import SystemMessage
from .memory import load_memory_sync, load_session_summaries_sync
from .tools import search_products, get_routine
from .state import AgentState


# ── helpers ────────────────────────────────────────────────────────────────

def _format_skin_profile(memory: dict) -> str:
    sp = memory.get("skin_profile", {})
    if not sp:
        return "No skin profile on file."
    lines = [
        f"Skin type: {sp.get('type', 'unknown')}",
        f"Skin tone: {sp.get('tone', 'unknown')}",
        f"Concerns: {', '.join(sp.get('concerns', [])) or 'none'}",
        f"Sensitivities: {', '.join(sp.get('sensitivities', [])) or 'none'}",
    ]
    prefs = memory.get("preferences", {})
    if prefs:
        lines.append(f"Prefers vegan: {prefs.get('vegan', False)}")
        lines.append(f"Prefers fragrance-free: {prefs.get('fragrance_free', False)}")
        budget = prefs.get("budget_range", [])
        if budget:
            lines.append(f"Budget range: ${budget[0]}–${budget[1]}")
    return "\n".join(lines)


def _format_products_for_prompt(products: list[dict]) -> str:
    if not products:
        return "No products found."
    lines = []
    for p in products:
        lines.append(
            f"- {p['name']} (${p.get('price', '?')}): {p.get('description', '')}. "
            f"Key ingredients: {', '.join(p.get('key_ingredients', []))}."
        )
    return "\n".join(lines)


# ── node factories (LLM injected via closure from build_graph) ─────────────

def make_memory_loader_node():
    """Load customer long-term memory from Lakebase."""
    def memory_loader_node(state: AgentState) -> dict:
        customer_id = state["customer_id"]
        memory = load_memory_sync(customer_id)
        summaries = load_session_summaries_sync(customer_id)
        if summaries:
            memory["session_summaries"] = summaries
        return {"memory": memory}
    return memory_loader_node


def make_intent_router_node():
    """Rule-based intent classification — no LLM call, fast."""
    def intent_router_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        msg = (last_msg.content if hasattr(last_msg, "content") else str(last_msg)).lower()

        shopper_keywords = ["show me", "browse", "find", "search", "compare", "all your", "list", "under $", "price"]
        coach_keywords = ["teach", "how to", "explain", "what is", "routine", "tip", "tutorial", "step by step", "difference between"]

        if any(kw in msg for kw in shopper_keywords):
            return {"intent": "shopper"}
        if any(kw in msg for kw in coach_keywords):
            return {"intent": "coach"}
        return {"intent": "advisor"}
    return intent_router_node


def make_advisor_node(llm):
    """Personalized recommendation node — calls LLM via ChatDatabricks."""
    def advisor_node(state: AgentState) -> dict:
        memory = state["memory"]
        prefs = memory.get("preferences", {})
        filters = {
            "vegan": prefs.get("vegan", False),
            "fragrance_free": prefs.get("fragrance_free", False),
        }
        if prefs.get("budget_range"):
            filters["max_price"] = prefs["budget_range"][1]

        affinities = memory.get("category_affinities", ["skincare"])
        last_msg = state["messages"][-1]
        query = last_msg.content if hasattr(last_msg, "content") else ""
        products = search_products(query=query, category=affinities[0], filters=filters)

        skin_summary = _format_skin_profile(memory)
        goals = memory.get("goals", [])
        product_list = _format_products_for_prompt(products)

        system_prompt = f"""You are Ava, Uphora's personal beauty advisor. Warm, expert, personalized.

Customer profile:
{skin_summary}
Beauty goals: {', '.join(goals) or 'general beauty'}

Relevant Uphora products:
{product_list}

Guidelines:
- Reference customer's skin type and concerns specifically.
- Recommend 1-3 products max, explain why each fits their profile.
- Never recommend products that conflict with their sensitivities.
- End with one follow-up question to deepen personalization.
- Tone: warm, expert, concise."""

        response = llm.invoke([SystemMessage(content=system_prompt)] + list(state["messages"]))
        return {"messages": [response], "products_found": products}
    return advisor_node


def make_shopper_node(llm):
    """Product discovery node — calls LLM via ChatDatabricks."""
    def shopper_node(state: AgentState) -> dict:
        memory = state["memory"]
        prefs = memory.get("preferences", {})
        filters = {"vegan": prefs.get("vegan", False), "fragrance_free": prefs.get("fragrance_free", False)}
        last_msg = state["messages"][-1]
        query = last_msg.content if hasattr(last_msg, "content") else ""
        products = search_products(query=query, category=None, filters=filters, limit=6)
        product_list = _format_products_for_prompt(products)

        system_prompt = f"""You are Ava, Uphora's shopping assistant.

Available products:
{product_list}

Guidelines:
- Present products with price and key benefit.
- Never invent products — only reference those listed.
- Offer to narrow by skin type, concern, or budget.
- Tone: helpful and efficient."""

        response = llm.invoke([SystemMessage(content=system_prompt)] + list(state["messages"]))
        return {"messages": [response], "products_found": products}
    return shopper_node


def make_coach_node(llm):
    """Beauty education and routine coaching node — calls LLM via ChatDatabricks."""
    def coach_node(state: AgentState) -> dict:
        memory = state["memory"]
        skin_summary = _format_skin_profile(memory)
        routine = get_routine(memory)
        am = " → ".join(routine["am"]) if routine["am"] else "not set"
        pm = " → ".join(routine["pm"]) if routine["pm"] else "not set"

        system_prompt = f"""You are Ava, Uphora's beauty coach.

Customer profile:
{skin_summary}

Current routines:
AM: {am}
PM: {pm}

Guidelines:
- Explain concepts clearly; assume the customer is learning.
- Connect advice to their skin profile.
- Suggest Uphora products only when directly applicable.
- Use numbered steps for routines.
- Tone: focused, warm, educational."""

        response = llm.invoke([SystemMessage(content=system_prompt)] + list(state["messages"]))
        return {"messages": [response], "products_found": []}
    return coach_node
