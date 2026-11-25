"""Success rates of cellbot1 on fresh tasks.

    uv run python -m cellbot.tasks.evaluate family --family herd --tasks 32     # a task family of the suite
    uv run python -m cellbot.tasks.evaluate push --tasks 32                    # pushing a cheese, at easier bars

family: a task succeeds if its checker scores at least 0.99 at the end and no cheese was eaten that
it did not ask for (as cellbot.steer.validate counts it). Tasks come from the test split: scenes and
wordings drawn like training's, never the same episodes.

push: two pushing probes. "direction" is the suite's "push the X cheese <direction>" task, counted at
how far the cheese moved in that direction (1, 2 and 4 cells; the suite's checker needs 4) without
being eaten. "corner" asks "push the X cheese into the <corner> corner", naming the corner nearest the
cheese, on scenes without walls; it succeeds if the cheese ends within 4 cells of that corner, uneaten.
The corner wording is not a training task.
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from cellbot.policy.inference import HUB_REPO, WEIGHTS, lm_sampler, load_policy, tokens
from cellbot.policy.model import PATCH
from cellbot.sim.world import COLOUR_NAMES, cheese_side, observe, step
from cellbot.tasks import suite as T

CORNERS = {"top left": (0, 0), "top right": (0, 1), "bottom left": (1, 0), "bottom right": (1, 1)}


def kind_of(task):
    """A short name for the instruction kind: the spec's main key (into, nudge, eat, box, ...)."""
    keys = [k for k in task.spec if k not in ("forbid", "order", "any_of")]
    return f"{task.family}.{keys[0]}" if keys else task.family


def success(score, unasked):
    return score >= 0.99 and not unasked


def family(args, sample, condition, device):
    rng = np.random.default_rng(args.seed)
    pairs = [T.sample_task(rng, "test", family=args.family) for _ in range(args.tasks)]
    outcomes = defaultdict(list)
    for k in range(0, len(pairs), args.batch):
        chunk = pairs[k : k + args.batch]
        results, _ = T.run(sample, condition, [s for s, _ in chunk], [t for _, t in chunk], args.steps, device)
        for (_, task), result in zip(chunk, results):
            outcomes[kind_of(task)].append(success(*result))
            outcomes[args.family].append(success(*result))
    return {kind: f"{sum(v)}/{len(v)}" for kind, v in sorted(outcomes.items())}


def direction_tasks(rng, n):
    out = []
    while len(out) < n:
        scene, task = T.sample_task(rng, "test", family="herd")
        if "nudge" in task.spec:
            out.append((scene, task))
    return out


def corner_tasks(rng, n):
    side = cheese_side(T.SIZE)
    far = T.SIZE - side
    out = []
    while len(out) < n:
        scene = T.sample_scene(rng, cheeses=(1, 3), boxes=(0, 1), wall_share=0.0)
        if scene is None:
            continue
        k = int(rng.integers(len(scene["cheeses"])))
        y, x = scene["cheeses"][k]
        name = min(CORNERS, key=lambda c: abs(y - CORNERS[c][0] * far) + abs(x - CORNERS[c][1] * far))
        target = (CORNERS[name][0] * far, CORNERS[name][1] * far)
        if max(abs(y - target[0]), abs(x - target[1])) <= 8:  # already close: nothing to show
            continue
        colour = COLOUR_NAMES[scene["cheese_colours"][k]]
        out.append((scene, {"text": f"push the {colour} cheese into the {name} corner", "slot": k, "target": target}))
    return out


@torch.no_grad()
def rollout(sample, condition, scenes, texts, steps, device):
    """(start, end) cheese positions (B, K, 2) and alive flags (B, K) of each scene."""
    world, _ = T.stack(scenes)
    world = world.to(device)
    start = world.cheese_pos.cpu().numpy().copy()
    cond = condition(texts)
    for _ in range(steps):
        _, field = sample(observe(world), cond)
        world, _ = step(world, field)
    return start, world.cheese_pos.cpu().numpy(), world.cheese_alive.cpu().numpy()


def push(args, sample, condition, device):
    rng = np.random.default_rng(args.seed)
    report = {}
    moved = []
    for k in range(0, args.tasks, args.batch):
        chunk = direction_tasks(rng, min(args.batch, args.tasks - k))
        start, end, alive = rollout(sample, condition, [s for s, _ in chunk], [t.text for _, t in chunk], args.steps, device)
        for i, (scene, task) in enumerate(chunk):
            c, d = task.spec["nudge"]
            slot = [COLOUR_NAMES[v] for v in scene["cheese_colours"]].index(c)
            ok = alive[i, slot] and alive[i].sum() == len(scene["cheeses"])  # nothing eaten
            dy, dx = T.DIRECTIONS[d]
            moved.append(float((end[i, slot] - start[i, slot]) @ np.array([dy, dx])) if ok else -1.0)
    report["direction"] = {f"moved >= {n} cells": f"{sum(m >= n for m in moved)}/{len(moved)}" for n in (1, 2, 4)}
    hits = []
    for k in range(0, args.tasks, args.batch):
        chunk = corner_tasks(rng, min(args.batch, args.tasks - k))
        start, end, alive = rollout(sample, condition, [s for s, _ in chunk], [t["text"] for _, t in chunk], args.steps, device)
        for i, (scene, task) in enumerate(chunk):
            slot, target = task["slot"], np.array(task["target"])
            near = np.abs(end[i, slot] - target).max() <= 4
            hits.append(bool(near and alive[i].sum() == len(scene["cheeses"])))
    report["corner"] = f"{sum(hits)}/{len(hits)}"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", choices=("family", "push"))
    parser.add_argument("--family", default="herd", choices=T.FAMILIES)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", default=HUB_REPO)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    policy, tokenizer = load_policy(args.policy, device, WEIGHTS)
    sample = lm_sampler(policy, 2.0, T.SIZE // PATCH)
    torch.manual_seed(args.seed)
    condition = lambda texts: tokens(tokenizer, texts, device)  # noqa: E731
    probe = family if args.probe == "family" else push
    print(json.dumps(probe(args, sample, condition, device)))


if __name__ == "__main__":
    main()
