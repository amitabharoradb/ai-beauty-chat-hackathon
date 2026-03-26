import json
import os
from databricks import sql as dbsql
from fastapi.responses import StreamingResponse
from mlflow.types.responses import ResponsesAgentRequest, ChatContext
from .core import create_router
from .models import CustomerOut, ChatRequest
from .agent.agent import AGENT
from .agent.memory import save_memory_sync
from .db import db_manager

router = create_router()

UC_CATALOG = os.getenv("UC_CATALOG", "amitabh_arora_catalog")
UC_SCHEMA = os.getenv("UC_SCHEMA", "uphora_hackathon")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "9465acf928ae5952")


def _fetch_demo_customers() -> list[dict]:
    """Fetch 10 demo customers from Unity Catalog via Databricks SQL."""
    from databricks.sdk import WorkspaceClient
    wc = WorkspaceClient()
    with dbsql.connect(
        server_hostname=wc.config.host.replace("https://", ""),
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        access_token=wc.config.token,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, name, skin_type, skin_tone
                FROM {UC_CATALOG}.{UC_SCHEMA}.customers
                WHERE id IN (
                    'cust_00001','cust_00002','cust_00003','cust_00004','cust_00005',
                    'cust_00006','cust_00007','cust_00008','cust_00009','cust_00010'
                )
                ORDER BY id
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


@router.get("/customers", response_model=list[CustomerOut], operation_id="listCustomers")
def list_customers():
    rows = _fetch_demo_customers()
    return [CustomerOut(**r) for r in rows]


@router.post("/chat", operation_id="chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    async def event_stream():
        # Build ResponsesAgentRequest (Mosaic AI Agent Framework)
        agent_request = ResponsesAgentRequest(
            input=[
                *[{"role": m.role, "content": m.content} for m in request.history],
                {"role": "user", "content": request.message},
            ],
            context=ChatContext(conversation_id=request.customer_id),
        )

        # Stream via UphoraBeautyAgent.predict_stream()
        for event in AGENT.predict_stream(agent_request):
            if event.type == "response.output_item.done" and hasattr(event.item, "text"):
                yield f"data: {json.dumps({'text': event.item.text})}\n\n"

        # Yield product cards
        products = AGENT.get_last_products()
        if products:
            yield f"data: {json.dumps({'products': products[:3]})}\n\n"

        # Persist memory delta
        recommended_ids = [p["id"] for p in products]
        if recommended_ids:
            memory = {}
            delta = {"product_history": {
                "recommended": recommended_ids,
                "liked": [],
                "disliked": [],
            }}
            save_memory_sync(request.customer_id, memory, delta)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
