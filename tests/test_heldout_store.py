"""The held-out-language store (experts.py --split held_out): its instructions are wordings and
compositions training never shows, on training's scenes."""

import json
from pathlib import Path

import numpy as np
import pytest

from cellbot.experts import attractor as E
from cellbot.sim.world import COLOUR_NAMES
from cellbot.tasks import suite as T

ROOT = Path(__file__).resolve().parents[2]
TRAINING_STORES = [ROOT / "data/vla/experts_g32", ROOT / "data/vla/experts_herd_g32"]
HELD_STORES = sorted(p for pattern in ("experts_heldout_g32*", "experts_newshapes_g32*") for p in (ROOT / "data/vla").glob(pattern) if (p / "done").exists())


def test_held_out_wordings_are_disjoint_from_training_wordings():
    relational = {f"logic.{form}" for form in T.HELD_OUT["relational"]}
    every_training = {w for key in T.WORDINGS if key not in relational for w in T.wordings(key, "")}
    for key in T.WORDINGS:
        if key in relational:  # templates no training form uses
            assert not set(T.WORDINGS[key]) & every_training, key
        else:  # the held-out fifth of the paraphrase pool
            held = set(T.wordings(key, "wording"))
            assert held and not held & every_training, key


def test_training_never_draws_held_out_compositions():
    rng = np.random.default_rng(0)
    held_forms = set(T.HELD_OUT["logic"]) | set(T.HELD_OUT["relational"])
    for _ in range(600):
        _, task = T.sample_task(rng, "train")
        assert task.held_out == ""
        if task.family == "logic":
            assert task.spec["form"] not in held_forms
        if task.family == "parallel":
            assert tuple(sorted(k for k, _ in task.spec["jobs"])) not in T.HELD_OUT["pairs"]
        if task.family == "herd":
            assert len(task.spec.get("into", ([None], None))[0]) == 1


def test_draw_held_draws_only_held_out_language_on_training_colours():
    rng = np.random.default_rng(1)
    pairs = E.draw_held(rng, 300)
    assert {t.held_out for _, t in pairs} <= set(E.HELD_LANGUAGE)
    assert "colour" not in E.HELD_LANGUAGE
    for scene, task in pairs:
        assert all(COLOUR_NAMES[c] in T.TRAIN_COLOURS for c in scene["cheese_colours"] + scene["box_colours"])
        if task.held_out == "relational":
            assert task.spec["form"] in T.HELD_OUT["relational"]


def test_unseen_shapes_are_new_names_and_new_masks():
    from cellbot.tasks.shapes import SHAPES, UNSEEN_SHAPES, shape_mask, unseen_shape_mask

    assert not set(UNSEEN_SHAPES) & set(SHAPES)
    seen = [shape_mask(s, T.SIZE) for s in SHAPES]
    for name in UNSEEN_SHAPES:
        mask = unseen_shape_mask(name, T.SIZE)
        assert 50 <= mask.sum() <= 150, name  # about as many squares as the training shapes
        assert all((mask != m).sum() > 20 for m in seen), name
    rng = np.random.default_rng(2)
    for _ in range(300):
        _, task = T.sample_task(rng, "train")
        assert not any(name in task.text for name in UNSEEN_SHAPES)
    _, task = T.sample_task(rng, "new_shapes")
    assert task.held_out == "new_shapes" and task.family == "formation"


def test_relational_forms_pick_the_referenced_cheese():
    mass = np.zeros((T.SIZE, T.SIZE), np.float32)
    mass[2:6, 2:6] = 1  # the swarm, top left
    scene = {
        "mass": mass,
        "walls": np.zeros((T.SIZE, T.SIZE), bool),
        "cheeses": [(8, 8), (26, 26), (14, 20)],
        "cheese_colours": [COLOUR_NAMES.index(c) for c in ("red", "blue", "green")],
        "boxes": [(24, 2, 6, 6)],  # bottom left: red nearest it, blue furthest
        "box_colours": [COLOUR_NAMES.index("white")],
        "runners": [],
    }
    want = {
        "nearest_self": ({"red"}, {"blue", "green"}),
        "except_furthest": ({"red", "green"}, {"blue"}),
        "furthest_box": ({"blue"}, {"red", "green"}),
        "except_nearest_box": ({"blue", "green"}, {"red"}),
    }
    seen = set()
    rng = np.random.default_rng(0)
    for _ in range(200):
        _, make = T.logic(rng, "relational")
        task = make(scene)
        form = task.spec["form"]
        assert (task.spec["eat"], task.spec["forbid"]) == want[form], form
        seen.add(form)
    assert seen == set(want)


@pytest.mark.skipif(not HELD_STORES or not all((p / "labels.jsonl").exists() for p in TRAINING_STORES), reason="needs the local stores under data/vla")
def test_held_out_store_texts_never_appear_in_training_stores():
    training = E.store_texts(TRAINING_STORES)
    for store in HELD_STORES:
        labels = [json.loads(line) for line in open(store / "labels.jsonl")]
        assert labels
        assert all(l["held_out"] in E.HELD_LANGUAGE + ("new_shapes",) for l in labels), store
        overlap = {l["text"] for l in labels} & training
        assert not overlap, (store.name, sorted(overlap)[:5])
