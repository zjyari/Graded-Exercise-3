# SCSE 26 — Graded Exercise #3 · RAG

RAG over the single-document university IT policy knowledge base, served by a local
**Qwen3-0.6B** model. The goal is to answer IT-support questions from the policy
document **without hallucinating** facts it does not contain.

## What was completed
- `policy_loader.py` — provided, **not modified**.
- `rag_tool.py` — builds a vector retriever over the policies:
  loads the document with `load_policy_documents()`, embeds each policy with a
  local `all-MiniLM-L6-v2` model, indexes them with Chroma.
- `rag.py` — the RAG pipeline: retrieves the top-k relevant policies, injects them
  into a system-prompted chat template, and generates a grounded answer with
  Qwen3-0.6B.
- `agent.py` — entry point that answers the graded question about the
  IT Service Desk phone number.

## Models
Both models are fetched from **ModelScope** (Hugging Face is unreachable when the
VPN is off) and stored under `models/`:
- `models/qwen3-0.6b` — `Qwen/Qwen3-0.6B`
- `models/all-MiniLM-L6-v2` — `sentence-transformers/all-MiniLM-L6-v2`

## Run
```bash
python agent.py
```

## Result
```
QUESTION: What exact phone number should I call for the university IT Service Desk?

ANSWER: The policy excerpts do not contain the exact information the user asked for
(a phone number). ... I do not have a phone number provided in the policy excerpts.
```

The model does **not** invent a phone number and does **not** produce an unrelated
answer — it correctly states the number is not available from the knowledge base.
`check_data.py` confirms the dataset contains **no** verified Service Desk phone number.
