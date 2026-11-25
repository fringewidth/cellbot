"""Instruction identification (vla/instruction_id.py) with a fake policy on a tiny expert store: no LM."""

import numpy as np
import torch

from cellbot.experts import attractor as E
from cellbot.tasks import suite as T
from cellbot.tasks.instruction_id import distractors, identify, instruction_losses, signature, summary


def test_signature_names_what_is_asked_not_how():
    assert signature("eat the red cheese") == signature("gobble up the red cheese") - {"up"} == {"red"}
    assert signature("spell HI") == signature("I'd like you to spell HI") == {"HI"}
    assert signature("go to the top left") == {"top", "left"}
    assert signature("eat the red cheese") != signature("eat the blue cheese")


def _labels():
    texts = [("words", "spell HI"), ("words", "please spell HI"), ("words", "spell OK"), ("words", "spell GO"), ("words", "spell UP"),
             ("logic", "eat everything but red and blue"), ("logic", "skip red and green, eat the rest"), ("pairs", "split up: go into the red box, and surround the blue cheese")]
    return [{"e": i, "store": 0, "family": "x", "held_out": item, "text": t} for i, (item, t) in enumerate(texts)]


def test_distractors_ask_for_something_else_from_the_same_item_and_are_fixed():
    labels = _labels()
    wrong = distractors(labels, 4, seed=0)
    assert wrong == distractors(labels, 4, seed=0)
    for l, w in zip(labels, wrong):
        assert len(set(w)) == 3 and l["text"] not in w
        assert all(signature(t) != signature(l["text"]) for t in w)
    assert set(wrong[0]) == {"spell OK", "spell GO", "spell UP"}  # same item; "please spell HI" means the same
    assert "spell HI" not in wrong[1]
    assert len(wrong[7]) == 3  # a one-episode item falls back to every item


def _store():
    rng = np.random.default_rng(0)
    pairs = [T.sample_task(rng, "train") for _ in range(3)]
    _, rec = E.play([s for s, _ in pairs], [t for _, t in pairs], steps=3)
    labels = [{"e": i, "store": 0, "family": t.family, "held_out": "", "text": f"true {i}"} for i, (_, t) in enumerate(pairs)]
    return rec, labels


def test_identification_follows_the_loss_of_the_true_text():
    store, labels = _store()
    wrong = [[f"wrong {i} {c}" for c in range(3)] for i in range(len(labels))]

    def reads_words(obs, actions, weights, texts):
        return torch.tensor([0.5 if t.startswith("true") else 1.0 for t in texts])

    losses = instruction_losses(reads_words, [store], labels, wrong, 2, 5, "cpu")
    correct, margin = identify(losses)
    assert losses.shape == (3, 4) and correct.tolist() == [1.0] * 3 and np.allclose(margin, 0.5)

    def ignores_words(obs, actions, weights, texts):
        return torch.ones(len(texts))

    correct, margin = identify(instruction_losses(ignores_words, [store], labels, wrong, 1, 5, "cpu"))
    assert correct.sum() == 0 and np.allclose(margin, 0.0)  # a tie is not an identification
    assert summary(correct, margin)[:3] == [0.0, 0.0, 0.0]


def test_scores_the_opening_scene():
    store, labels = _store()
    wrong = [["a", "b", "c"]] * len(labels)

    def robots(obs, actions, weights, texts):
        return obs[:, 0].sum((1, 2)).float().cpu()

    losses = instruction_losses(robots, [store], labels, wrong, 1, 4, "cpu")
    expected = store["mass"][:, 0].float().sum((1, 2)).numpy()
    assert np.allclose(losses, expected[:, None])
