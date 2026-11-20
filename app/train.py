"""Continue CPU pretraining on synthetic agricultural event timelines."""
from dataclasses import asdict
from pathlib import Path

from app.core import Bigram, split_fields
from app.llm import continue_pretraining, encode_sequence, extend_tokenizer, load_base
from app.storage import dataset_digest, load_fields, save_model, write_json


def train_run(
    data: Path, output: Path, results: Path, base: str, seed: int, epochs: int,
    batch_size: int, learning_rate: float, threads: int,
) -> dict:
    """Train the LM and fit the counting baseline using the same training fields."""
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    fields = split_fields(load_fields(data), seed)
    model, tokenizer = load_base(base)
    ids = extend_tokenizer(model, tokenizer)
    train, val = ([encode_sequence(field.sequence(), tokenizer, ids) for field in fields[split]]
                  for split in ("train", "validation"))
    if max(map(len, train + val)) > model.config.max_position_embeddings:
        raise ValueError("A field exceeds the model context limit.")
    history = continue_pretraining(
        model, train, val, pad=tokenizer.pad_token_id, epochs=epochs,
        batch_size=batch_size, learning_rate=learning_rate, seed=seed,
    )
    metadata = {"seed": seed, "base": base, "dataset_sha256": dataset_digest(data),
                "splits": {key: len(value) for key, value in fields.items()}}
    save_model(output, model, tokenizer, metadata)
    write_json(output / "bigram.json", asdict(Bigram.fit(fields["train"])))
    report = {**metadata, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
              "history": history, "parameter_count": sum(p.numel() for p in model.parameters())}
    write_json(results / "training.json", report)
    return report
