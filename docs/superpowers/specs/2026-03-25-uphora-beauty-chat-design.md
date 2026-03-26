# Uphora AI Beauty Chat — PRD

## Context

Uphora is a fictitious high-end beauty brand. This project builds a conversational AI chatbot that acts as a personal beauty advisor, product shopper, and beauty coach — all in one. The chatbot is deployed on Databricks Apps and powered by a LangGraph agent with persistent memory in Lakebase Autoscaling.

---

## Goals

- Demonstrate a production-grade AI agent on Databricks (LangGraph + Lakebase + Apps)
- Show personalized, memory-driven beauty recommendations
- Use realistic fake data (customers, products) to simulate a real retail scenario

---

## Functional Requirements

### Chatbot Persona
All-in-One Hybrid — the bot adapts per message:
- **Advisor**: personalized skin/routine recommendations based on customer profile
- **Shopper**: product discovery and comparison from the Uphora catalog
- **Coach**: beauty education, routine building, tips and techniques

### Customer Identity (Demo Mode)
- Dropdown selector of 5–10 demo customers (name + skin type shown)
- Selecting a customer sets `customer_id` for the session — no real auth
- Each demo customer has a pre-seeded long-term memory profile

### Purchase Flow
- Recommend only — no checkout in chat
- Product cards displayed inline with name, price, key benefits
- "View on Uphora" link on each card (external link placeholder)

### Memory
**Short-term (LangGraph state, in-session):**
- Full conversation history
- Current intent (advisor / shopper / coach)
- Products surfaced this session
- Active skin/goal context from current conversation

**Long-term (Lakebase Autoscaling, persisted across sessions):**
| Field | Purpose |
|-------|---------|
| Skin profile (type, tone, concerns, sensitivities) | Personalize every recommendation |
| Beauty goals (anti-aging, glow, acne control, etc.) | Drive advisor + coach responses |
| Preferences (vegan, fragrance-free, budget range) | Filter product suggestions |
| Product history (recommended, liked, disliked) | Avoid repeats, learn taste |
| Saved routines (AM/PM steps) | Coach node retrieves and evolves |
| Category affinities (skincare > makeup > haircare) | Prioritize relevant category |
| Session summaries (last 5, compressed) | Context without full history replay |

**Memory lifecycle:**
1. Session start → Memory Loader reads Lakebase for `customer_id`
2. During session → LangGraph state holds everything in-memory
3. Each turn end → Memory Writer upserts deltas to Lakebase

---

## Non-Functional Requirements

| Requirement | Decision |
|------------|---------|
| Agent framework | LangGraph (Python) |
| Agent deployment | Databricks Apps (apx — React + FastAPI) |
| Memory store | Databricks Lakebase Autoscaling (PostgreSQL) |
| Product/customer data | Unity Catalog — catalog `amitabh_arora_catalog`, schema `uphora_hackathon` |
| LLM | Claude Sonnet 4.6 via Databricks Foundation Model API — `w.serving_endpoints.query(name="databricks-claude-sonnet-4-6", stream=True)` using `databricks-sdk`, no extra package needed |
| Fake data generation | Spark + Faker |

---

## Data Design

### Unity Catalog — Delta Tables

| Table | Rows | Key Columns |
|-------|------|-------------|
| `customers` | 10,000 | id, name, email, age, skin_type, skin_tone, concerns |
| `categories` | 3 | id, name |
| `products` | 5–15 per category (~30 total) | id, category_id, name, description, price, key_ingredients, benefits, tags |
| `customer_products` | ~50K | customer_id, product_id, interaction_type (purchased/viewed/liked) |

**Categories (Sephora-inspired):**
- **Skincare**: cleansers, serums, moisturizers, SPF, eye cream, toner, masks
- **Makeup**: foundation, concealer, mascara, lipstick, blush, highlighter, eyeshadow
- **Haircare**: shampoo, conditioner, hair mask, scalp serum, styling cream, dry shampoo

### Lakebase Autoscaling — PostgreSQL Tables

| Table | Key Columns |
|-------|------------|
| `customer_memory` | customer_id, skin_profile (JSON), goals (JSON), preferences (JSON), routines (JSON), product_history (JSON), updated_at |
| `session_summaries` | customer_id, session_id, summary_text, created_at |

---

## System Architecture

```
┌─────────────────────────────────────────────┐
│           Databricks App (apx)              │
│     React frontend + FastAPI backend        │
│     Warm luxury UI (cream/gold/charcoal)    │
│     Customer dropdown (5-10 demo users)     │
└──────────────────┬──────────────────────────┘
                   │ customer_id + message
┌──────────────────▼──────────────────────────┐
│         LangGraph Agent (Python)            │
│                                             │
│  memory_loader → intent_router              │
│       ↓              ↓                      │
│  advisor_node  shopper_node  coach_node     │
│       ↓                                     │
│  memory_writer                              │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│   Lakebase Autoscaling (long-term memory)   │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│   Unity Catalog (customers + products)      │
└─────────────────────────────────────────────┘
```

## Agent Tools

| Tool | Description |
|------|------------|
| `search_products(query, category, filters)` | Queries Unity Catalog product tables |
| `get_customer_memory(customer_id)` | Reads Lakebase long-term memory |
| `update_memory(customer_id, delta)` | Upserts deltas to Lakebase |
| `get_routine(customer_id)` | Retrieves saved AM/PM routines |

---

## UI Design

**Style**: Warm luxury (Sephora-inspired)
- Background: cream (`#faf8f5`)
- Accents: gold (`#c9a96e`)
- Text: charcoal (`#2d2d2d`)
- Typography: clean sans-serif, spaced letter tracking
- Chat bubbles: white with warm border (bot), charcoal (user)
- Product cards: white cards with warm border, gold price

**Key screens:**
1. Customer selector (dropdown, shown on load)
2. Chat view (full-width, product cards inline)
3. Sidebar: customer profile summary + saved routine (phase 2)

---

## Verification Plan

1. **Data generation**: run Spark notebook → verify 10K customers, ~30 products in UC tables
2. **Lakebase**: create `customer_memory` + `session_summaries` tables, seed 10 demo customers
3. **Agent**: test LangGraph locally — send message, verify memory load → intent route → tool call → memory write
4. **App**: run apx dev server, verify customer dropdown loads, chat sends messages, product cards render
5. **End-to-end**: select customer → chat → switch customer → verify memory isolation
