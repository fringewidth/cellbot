"""Instruction identification: does a policy read the words? For each demonstration, the expert's
first move(s) from the episode's opening scene are scored under the true instruction and under k - 1
wrong ones (other episodes' instructions, from the same held-out item where it has enough); the policy
identifies the instruction when the true one gives the lowest loss. Chance is 1 / k. At step 0 nothing
has moved yet, so only the words say what the swarm is about to do.

Wrong instructions must ask for something else, not reword the true one: a candidate is skipped when
it names the same things (signature: colours, capitalised letters and words, shapes, region and
direction words). This is conservative (two different relational forms naming nothing are both
skipped), but not exact: a filler word such as "up" in "eat up" counts as a direction, so a few wrong
instructions may still mean the same; that lowers every checkpoint's ceiling alike.

Used to score whether the policy can tell instructions apart.
"""

import re

import numpy as np
import torch

from cellbot.sim.world import COLOUR_NAMES
from cellbot.tasks.suite import DIRECTIONS, REGIONS
from cellbot.training.data import batch_from

SHAPE_NAMES = {"ring", "circle", "square", "plus", "x", "horizontal", "vertical", "diagonal", "triangle", "heart"}
SLOT_WORDS = set(COLOUR_NAMES) | SHAPE_NAMES | {w for phrase in list(REGIONS) + list(DIRECTIONS) for w in phrase.split()} - {"and"}


def signature(text):
    """What an instruction names: its slot words (lowercase) and capitalised letters or words."""
    capitals = {w for w in re.findall(r"\b[A-Z]{1,4}\b", text) if w != "I"}
    return frozenset(capitals | (set(re.findall(r"[a-z]+", text.lower())) & SLOT_WORDS))


def item_of(label):
    return label.get("held_out") or label["family"]


def distractors(labels, k, seed=0):
    """Per label, k - 1 distinct wrong instruction texts (fixed by seed): drawn from the texts of the
    same item whose signature differs, or from all texts if the item has fewer than k - 1 of those."""
    rng = np.random.default_rng(seed)
    texts = sorted({l["text"] for l in labels})
    sig = {t: signature(t) for t in texts}
    by_item = {}
    for l in labels:
        by_item.setdefault(item_of(l), set()).add(l["text"])
    by_item = {i: sorted(ts) for i, ts in by_item.items()}
    out = []
    for l in labels:
        mine = sig[l["text"]]
        pool = [t for t in by_item[item_of(l)] if sig[t] != mine]
        if len(pool) < k - 1:
            pool = [t for t in texts if sig[t] != mine]
        out.append([str(t) for t in rng.choice(pool, size=k - 1, replace=False)])
    return out


@torch.no_grad()
def instruction_losses(loss_fn, stores, labels, wrong, first_steps, batch, device):
    """(N, k) loss of each label's expert moves at steps 0 .. first_steps - 1 (each from the scene at
    that step), averaged over those steps, under its true text (column 0) and its wrong ones.
    loss_fn(obs, actions, weights, texts) -> (B,) per-sample loss. labels carry "store" and "e"."""
    k = 1 + len(wrong[0])
    total = torch.zeros(len(labels), k)
    rows = [(i, t, c) for i in range(len(labels)) for t in range(first_steps) for c in range(k)]
    for start in range(0, len(rows), batch):
        part = rows[start : start + batch]
        for s, store in enumerate(stores):
            mine = [(i, t, c) for i, t, c in part if labels[i]["store"] == s]
            if not mine:
                continue
            pick = torch.tensor([[labels[i]["e"], t, t + 1] for i, t, _ in mine])  # a one-step window: the step itself
            obs, _, actions, weights, _ = batch_from(store, pick, torch.Generator().manual_seed(0), device)
            texts = [labels[i]["text"] if c == 0 else wrong[i][c - 1] for i, _, c in mine]
            for (i, _, c), v in zip(mine, loss_fn(obs, actions, weights, texts).tolist()):
                total[i, c] += v
    return (total / first_steps).numpy()


def identify(losses):
    """Per episode: correct (the true text's loss strictly lowest) and margin (mean wrong - true)."""
    true, wrong = losses[:, 0], losses[:, 1:]
    return (true < wrong.min(1)).astype(np.float64), wrong.mean(1) - true


def summary(correct, margin):
    """[accuracy, its binomial standard error, mean margin, its standard error, n]."""
    n = len(correct)
    acc = float(correct.mean())
    se = float(margin.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return [round(acc, 4), round(float(np.sqrt(acc * (1 - acc) / n)), 4), round(float(margin.mean()), 4), round(se, 4), n]
