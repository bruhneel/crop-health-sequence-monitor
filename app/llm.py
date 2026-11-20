"""One chat call that works with Ollama, Groq or Gemini: change three settings in .env, not the code."""
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    import torch
    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

import httpx


def _load_env() -> None:
    """Read KEY=VALUE lines from .env (if there is one) without overriding real environment variables."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def settings() -> tuple[str, str, str]:
    _load_env()
    return (os.environ.get("LLM_BASE_URL") or "http://localhost:11434/v1",
            os.environ.get("LLM_API_KEY", ""), os.environ.get("LLM_MODEL", ""))


def chat(messages: list[dict], **options) -> str:
    """Send chat messages, get the reply text. `options` go straight to the API (temperature, max_tokens...)."""
    base, key, model = settings()
    if not model:
        raise RuntimeError("Set LLM_MODEL in .env (see .env.example).")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = httpx.post(f"{base.rstrip('/')}/chat/completions", headers=headers, timeout=120,
                   json={"model": model, "messages": messages, **options})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


DEFAULT_BASE = "distilbert/distilgpt2"


def load_base(name: str = DEFAULT_BASE) -> tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
    """Load the public pretrained model and tokenizer anonymously on CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, token=False)
    model = AutoModelForCausalLM.from_pretrained(name, token=False, use_safetensors=True)
    return model, tokenizer


def event_token(item: str) -> str:
    return f"<|{item}|>"


def event_ids(tokenizer: "PreTrainedTokenizerBase") -> dict[str, int]:
    """Resolve the distinct single-token IDs of the agricultural vocabulary."""
    from app.core import ITEMS

    ids = {}
    for item in ITEMS:
        token = event_token(item)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if token not in tokenizer.all_special_tokens or len(encoded) != 1:
            raise ValueError(f"Checkpoint tokenizer is missing {token}.")
        ids[item] = encoded[0]
    if len(set(ids.values())) != len(ids):
        raise ValueError("Event tokens must map to distinct IDs.")
    return ids


def extend_tokenizer(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizerBase") -> dict[str, int]:
    """Create one token per event and initialize it from its ordinary words."""
    import torch

    from app.core import ITEMS

    old_size = len(tokenizer)
    if model.get_input_embeddings().num_embeddings != old_size:
        raise ValueError("Base tokenizer and model vocabulary sizes differ.")
    original = {item: tokenizer.encode(" " + item.replace("_", " "), add_special_tokens=False)
                for item in ITEMS}
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [event_token(item) for item in ITEMS]},
        replace_additional_special_tokens=False,
    )
    if tokenizer.eos_token_id is None or tokenizer.bos_token_id is None:
        raise ValueError("The model must have beginning and end tokens.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = event_ids(tokenizer)
    if len(tokenizer) != old_size:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        weights = model.get_input_embeddings().weight
        with torch.no_grad():
            for item, token_id in ids.items():
                if token_id >= old_size:
                    weights[token_id].copy_(weights[original[item]].mean(0))
        model.tie_weights()
    model.config.pad_token_id = tokenizer.pad_token_id
    return ids


def encode_sequence(
    sequence: "Sequence[str]", tokenizer: "PreTrainedTokenizerBase",
    ids: dict[str, int], *, complete: bool = True,
) -> list[int]:
    """Represent the ordered textual event tokens without interspersed separator IDs."""
    encoded = [tokenizer.bos_token_id, *(ids[item] for item in sequence)]
    return encoded + [tokenizer.eos_token_id] if complete else encoded


def _packed(rows: "Sequence[list[int]]", pad: int) -> tuple["Tensor", "Tensor"]:
    import torch

    width = max(map(len, rows))
    ids = torch.tensor([row + [pad] * (width - len(row)) for row in rows])
    mask = torch.arange(width).unsqueeze(0) < torch.tensor([len(row) for row in rows]).unsqueeze(1)
    return ids, mask.long()


def _epoch(
    model: "PreTrainedModel", rows: "Sequence[list[int]]", batch_size: int, pad: int,
    optimizer: "torch.optim.Optimizer | None" = None,
) -> float:
    import torch

    model.train(optimizer is not None)
    total_loss = 0.0
    total_tokens = 0
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0, len(rows), batch_size):
            tokens, mask = _packed(rows[start:start + batch_size], pad)
            labels = tokens.masked_fill(mask == 0, -100)
            loss = model(input_ids=tokens, attention_mask=mask, labels=labels, use_cache=False).loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = int(mask[:, 1:].sum())
            total_loss += loss.item() * count
            total_tokens += count
    return total_loss / total_tokens


def continue_pretraining(
    model: "PreTrainedModel", train: "Sequence[list[int]]", val: "Sequence[list[int]]", *,
    pad: int, epochs: int, batch_size: int, learning_rate: float, seed: int, log: bool = True,
) -> list[dict[str, float | int]]:
    """Run standard shifted causal-LM cross-entropy over all non-padding targets."""
    import math
    import random

    import torch

    if min(len(train), len(val), epochs, batch_size) < 1:
        raise ValueError("Training requires train and validation rows, epochs, and batch size.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite.")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = [{"epoch": 0, "train_loss": _epoch(model, train, batch_size, pad),
                "val_loss": _epoch(model, val, batch_size, pad)}]
    if log:
        print(f'epoch  0  train {history[0]["train_loss"]:.4f}  val {history[0]["val_loss"]:.4f}', flush=True)
    for epoch in range(1, epochs + 1):
        shuffled = list(train)
        rng.shuffle(shuffled)
        training = _epoch(model, shuffled, batch_size, pad, optimizer)
        validation = _epoch(model, val, batch_size, pad)
        history.append({"epoch": epoch, "train_loss": training, "val_loss": validation})
        if log:
            print(f"epoch {epoch:2d}  train {training:.4f}  val {validation:.4f}", flush=True)
    return history
