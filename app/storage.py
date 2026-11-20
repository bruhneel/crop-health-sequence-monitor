"""A tiny key-value store in one JSON file (no server, easy to read). Values can be anything JSON can hold."""
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core import FieldHistory
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class Store:
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.environ.get("DATA_PATH", "data/app.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}")

    def _read(self) -> dict:
        return json.loads(self.path.read_text() or "{}")

    def put(self, key: str, value) -> None:
        data = self._read()
        data[key] = value
        self.path.write_text(json.dumps(data, indent=2))

    def get(self, key: str, default=None):
        return self._read().get(key, default)

    def all(self) -> dict:
        return dict(sorted(self._read().items()))


def write_fields(path: Path, fields: list["FieldHistory"]) -> None:
    """Write synthetic field histories as UTF-8 JSON Lines."""
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for field in fields:
            target.write(json.dumps(asdict(field), separators=(",", ":")) + "\n")


def load_fields(path: Path = Path("data/sequences.jsonl")) -> list["FieldHistory"]:
    """Load and validate every field and enforce unique field IDs."""
    from app.core import FieldHistory

    fields = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("field_id"), str) or not isinstance(row.get("seasons"), list):
            raise ValueError(f"Invalid field record on line {number}.")
        if any(not isinstance(season, list) or any(not isinstance(item, str) for item in season)
               for season in row["seasons"]):
            raise ValueError(f"Invalid season on line {number}.")
        field = FieldHistory(row["field_id"], tuple(tuple(events) for events in row["seasons"]))
        field.validate()
        if field.field_id in seen:
            raise ValueError(f"Duplicate field ID: {field.field_id}.")
        seen.add(field.field_id)
        fields.append(field)
    return fields


def write_json(path: Path, report: dict) -> None:
    """Write a deterministic JSON report for later inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report(directory: Path, name: str, report: dict, markdown: str) -> None:
    """Write both output formats under the same stem."""
    write_json(directory / f"{name}.json", report)
    (directory / f"{name}.md").write_text(markdown, encoding="utf-8")


def dataset_digest(path: Path) -> str:
    """Fingerprint the input dataset used to train the checkpoint."""
    from hashlib import sha256

    return sha256(path.read_bytes()).hexdigest()


def save_model(
    path: Path, model: "PreTrainedModel", tokenizer: "PreTrainedTokenizerBase", metadata: dict
) -> None:
    """Save a local safetensors checkpoint, tokenizer, and dataset metadata."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    write_json(path / "monitor.json", metadata)


def load_model(path: Path) -> tuple["PreTrainedModel", "PreTrainedTokenizerBase", dict]:
    """Load the saved model locally with a clear missing-checkpoint message."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (path / "monitor.json").is_file():
        raise FileNotFoundError("No trained model found. Run python -m app.cli train first.")
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model.eval(), tokenizer, json.loads((path / "monitor.json").read_text(encoding="utf-8"))
