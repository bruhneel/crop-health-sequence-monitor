"""Construct a tiny checkpoint and tokenizer without downloading weights."""
import pytest


@pytest.fixture
def tiny_pair():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from app.core import ITEMS
    from app.llm import extend_tokenizer

    torch.set_num_threads(1)
    torch.manual_seed(42)
    backend = Tokenizer(WordLevel(unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.train_from_iterator([item.replace("_", " ") for item in ITEMS],
                                WordLevelTrainer(special_tokens=["[UNK]", "[BOS]", "[EOS]"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       bos_token="[BOS]", eos_token="[EOS]")
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=64, n_embd=24, n_layer=1, n_head=4,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    ))
    extend_tokenizer(model, tokenizer)
    return model, tokenizer
