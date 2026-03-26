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
            agent_request = ResponsesAgentRequest(
                input=[
                    *[{"role": m.role, "content": m.content} for m in request.history],
                    {"role": "user", "content": request.message},
                ],
                context=ChatContext(conversation_id=request.customer_id),
            )

            # Run sync agent in thread pool so it doesn't block the event loop
            loop = asyncio.get_event_loop()

            def run_agent():
                results = []
                try:
                    for event in AGENT.predict_stream(agent_request):
                        if event.type == "response.output_item.done" and hasattr(event.item, "text"):
                            results.append(("text", event.item.text))
                    results.append(("products", AGENT.get_last_products()))
                except Exception as e:
                    import traceback
                    print(f"[agent] error: {traceback.format_exc()}")
                    results.append(("error", str(e)))
                return results

            results = await loop.run_in_executor(_executor, run_agent)

            for kind, value in results:
                if kind == "text":
                    yield f"data: {json.dumps({'text': value})}\n\n"
                elif kind == "products" and value:
                    yield f"data: {json.dumps({'products': value[:3]})}\n\n"
                elif kind == "error":
                    yield f"data: {json.dumps({'error': value})}\n\n"

        except Exception as e:
            import traceback
            print(f"[router] /api/chat error: {traceback.format_exc()}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
