### COMPLETE THE CODE ###


## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
from policy_loader import load_policy_documents

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_tool import build_retriever

# Qwen3-0.6B served locally (fetched from ModelScope so Hugging Face is not required)
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = str(MODELS_DIR / "qwen3-0.6b")

SYSTEM_PROMPT = (
    "You are a university IT support assistant. Answer the user's question "
    "using ONLY the policy excerpts provided below. If the excerpts do not "
    "contain the exact information the user asked for (for example, a phone "
    "number), say that you do not have that information instead of guessing, "
    "inventing a number, or giving unrelated advice. Keep the answer concise."
)

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
        if torch.cuda.is_available():
            _model = _model.to("cuda")
        _model.eval()
    return _model, _tokenizer


def _build_prompt(question: str, context: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Policy excerpts:\n\n{context}\n\nQuestion: {question}",
        },
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        # Older template without a thinking-mode switch
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def answer_question(question: str, retriever=None) -> str:
    """Retrieve the relevant policy excerpts and generate a grounded answer."""
    if retriever is None:
        retriever = build_retriever()

    model, tokenizer = _load_model()

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = _build_prompt(question, context, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
