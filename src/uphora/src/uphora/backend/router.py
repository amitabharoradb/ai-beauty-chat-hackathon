import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from databricks import sql as dbsql
from fastapi.responses import StreamingResponse
from mlflow.types.responses import ResponsesAgentRequest, ChatContext
from .core import create_router
from .models import CustomerOut, ChatRequest
from .agent.agent import AGENT
from .agent.memory import save_memory_sync
from .db import db_manager

_executor = ThreadPoolExecutor(max_workers=4)

router = create_router()

UC_CATALOG = os.getenv("UC_CATALOG", "amitabh_arora_catalog")
UC_SCHEMA = os.getenv("UC_SCHEMA", "uphora_hackathon")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "9465acf928ae5952")


# Hardcoded demo customers — avoids SQL warehouse cold-start on page load
_DEMO_CUSTOMERS = [
    {"id": "cust_00001", "name": "Ava Chen",        "skin_type": "oily",        "skin_tone": "medium"},
    {"id": "cust_00002", "name": "Sofia Rossi",      "skin_type": "dry",         "skin_tone": "fair"},
    {"id": "cust_00003", "name": "Maya Johnson",     "skin_type": "combination", "skin_tone": "light"},
    {"id": "cust_00004", "name": "Priya Patel",      "skin_type": "sensitive",   "skin_tone": "deep"},
    {"id": "cust_00005", "name": "Zara Williams",    "skin_type": "normal",      "skin_tone": "tan"},
    {"id": "cust_00006", "name": "Leila Hassan",     "skin_type": "oily",        "skin_tone": "medium"},
    {"id": "cust_00007", "name": "Emma Tanaka",      "skin_type": "dry",         "skin_tone": "fair"},
    {"id": "cust_00008", "name": "Camille Dubois",   "skin_type": "combination", "skin_tone": "light"},
    {"id": "cust_00009", "name": "Amara Okafor",     "skin_type": "normal",      "skin_tone": "deep"},
    {"id": "cust_00010", "name": "Isabella Martins", "skin_type": "sensitive",   "skin_tone": "medium"},
]

def _fetch_demo_customers() -> list[dict]:
    return _DEMO_CUSTOMERS


@router.get("/customers", response_model=list[CustomerOut], operation_id="listCustomers")
def list_customers():
    rows = _fetch_demo_customers()
    return [CustomerOut(**r) for r in rows]


@router.post("/chat", operation_id="chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    async def event_stream():
        try:
            loop = asyncio.get_event_loop()

            def run_agent():
                try:
                    from mlflow.types.responses import ResponsesAgentRequest, ChatContext
                    agent_request = ResponsesAgentRequest(
                        input=[
                            *[{"role": m.role, "content": m.content} for m in request.history],
                            {"role": "user", "content": request.message},
                        ],
                        context=ChatContext(conversation_id=request.customer_id),
                    )
                    # predict() uses graph.invoke() internally — no streaming complexity
                    response = AGENT.predict(agent_request)
                    # Extract text from output items
                    text = ""
                    for item in response.output:
                        t = getattr(item, "text", None) or getattr(item, "content", None)
                        if t:
                            text += t
                    products = AGENT.get_last_products()
                    return ("ok", text, products)
                except Exception as e:
                    import traceback
                    return ("error", f"{traceback.format_exc()}", [])

            kind, text, products = await loop.run_in_executor(_executor, run_agent)

            if kind == "ok":
                yield f"data: {json.dumps({'text': text})}\n\n"
                if products:
                    yield f"data: {json.dumps({'products': products[:3]})}\n\n"
            else:
                yield f"data: {json.dumps({'error': text})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
