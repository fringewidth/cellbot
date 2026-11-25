"""Loading cellbot1 and sampling moves from it.

    from cellbot.policy.inference import load_policy
    policy, tokenizer = load_policy("fringewidth/cellbot1", torch.device("cpu"))

A source is a Hugging Face model repo or a local directory holding config.json and the checkpoint
(the LoRA and readout weights only; the base LM comes from its own repo, named in config.json).
"""

import json
from pathlib import Path

import torch

from cellbot.policy.actions import decode_field
from cellbot.policy.model import PATCH, LMPolicy

HUB_REPO = "fringewidth/cellbot1"
WEIGHTS = "cellbot1.safetensors"
MAX_TEXT_TOKENS = 96


def fetch(source, checkpoint):
    """(config.json path, checkpoint path) of a local directory, or downloaded from a Hub repo."""
    local = Path(source)
    if local.is_file():
        local = local.parent
    if local.is_dir():
        return local / "config.json", local / checkpoint
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(source, "config.json")), Path(hf_hub_download(source, checkpoint))


def read_weights(path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path)
    saved = torch.load(path, map_location="cpu")
    return saved.get("weights", saved)


def load_policy(source=HUB_REPO, device=torch.device("cpu"), checkpoint=WEIGHTS):
    """(policy, tokenizer). On CUDA the frozen LM weights are fp16, as in training; elsewhere fp32."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path, weights_path = fetch(source, checkpoint)
    run = json.loads(config_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(run["model"])
    tokenizer.padding_side = "right"
    lm = AutoModelForCausalLM.from_pretrained(run["model"], dtype=torch.float16 if device.type == "cuda" else torch.float32)
    policy = LMPolicy(lm, lora_rank=run["lora_rank"], freeze_embeddings=run.get("freeze_embeddings", True))
    weights = read_weights(weights_path)
    missing = [n for n, p in policy.named_parameters() if p.requires_grad and n not in weights]
    assert not missing, f"checkpoint lacks trainable weights: {missing[:5]}"
    policy.load_state_dict({k: v.float() for k, v in weights.items()}, strict=False)
    return policy.to(device).eval(), tokenizer


def tokens(tokenizer, texts, device):
    """Token ids and attention mask (B, L) of a batch of instructions."""
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TEXT_TOKENS)
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def lm_probs_sampler(policy, guidance, n):
    """sample(obs, (ids, mask)) -> (moves, field, probs): one sampled move per square, the move field it
    decodes to, and the probabilities (after classifier-free guidance) as (B, G, G, 9) in grid order.
    n is the number of patches per grid side."""

    def sample(obs, cond):
        ids, mask = cond
        if guidance == 1.0:
            out = policy.logits(obs, text_ids=ids, text_mask=mask)
        else:  # the instruction and the null instruction (the scene alone) in one forward
            b = len(obs)
            null = torch.arange(2 * b, device=obs.device) >= b
            both = policy.logits(torch.cat([obs, obs]), text_ids=torch.cat([ids, ids]), text_mask=torch.cat([mask, mask]), null=null)
            out = both[b:] + guidance * (both[:b] - both[b:])
        moves = torch.distributions.Categorical(logits=out.float()).sample()
        b, g = len(obs), obs.shape[-1]
        codes = moves.view(b, n, n, PATCH, PATCH).permute(0, 1, 3, 2, 4).reshape(b, g, g)
        probs = torch.softmax(out.float(), dim=-1).view(b, n, n, PATCH, PATCH, -1).permute(0, 1, 3, 2, 4, 5).reshape(b, g, g, -1)
        return moves, decode_field(codes, obs.device), probs

    return sample


def lm_sampler(policy, guidance, n):
    """As lm_probs_sampler, without the probabilities: sample(obs, cond) -> (moves, field), the
    signature tasks.suite.run expects."""
    sample_with_probs = lm_probs_sampler(policy, guidance, n)

    def sample(obs, cond):
        moves, field, _ = sample_with_probs(obs, cond)
        return moves, field

    return sample
