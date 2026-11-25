"""Recordings of cellbot1 at work, the input of the demo video (cellbot.demo.render).

Each instruction in INSTRUCTIONS is rolled out on fresh scenes in one batch, either on the grid the
policy was trained in (cellbot.sim.world) or in MuJoCo (cellbot.sim.mujoco_world, snapped to the
grid for the policy). The policy's per-square move probabilities are sampled every step. A command
is scored by its task's checker (tasks.suite.score) and ends once met plus a few holding steps; a
command without a checker (free text in the Space) runs a fixed number of steps.

    uv run python -m cellbot.demo.record --physics mujoco --only 0,2,3,5,7,16 --rollouts 8 --out runs/demo/mujoco.json
    uv run python -m cellbot.demo.record --physics grid --only 0,2,3,5,7,16 --rollouts 8 --out runs/demo/grid.json

Format (per episode, per step t: the state at t and the policy's output at that state; the sampled
moves of step t take the world to step t + 1):
  robots   flat [row, col, code, ...] ordered by robot id (ids are stable across steps)
  cheeses  [[row, col, alive], ...] by cheese slot; eaten / pushed: slots eaten / pushed on the way to t
  field    base64 int8 (2, 32, 32): expected vector sum_a p_a * CODE_VECTORS[a], (x, y), times 127
  pmax     base64 uint8 (32, 32): max_a p_a times 255
  obs      per channel an index into the episode's obs_blobs (base64 uint8 32x32, value times 255)
  path     MuJoCo only: base64 float16 (frames, robots + cheeses, 2), the continuous positions over the step
"""

import argparse
import base64
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from cellbot.policy.actions import CODE_VECTORS, encode_field
from cellbot.policy.inference import HUB_REPO, WEIGHTS, lm_probs_sampler, load_policy, tokens
from cellbot.policy.model import PATCH
from cellbot.sim.mujoco_world import FRAMES_PER_STEP, Batch
from cellbot.sim.world import COLOUR_NAMES, COLOUR_RGB, DIRS, STAY, cheese_side, eat_cheeses, observe, step
from cellbot.steer import scene_of
from cellbot.tasks import suite as T
from cellbot.tasks.shapes import shape_mask, text_mask

CHANNELS = ("mass", "wall", "cheese mask", "cheese R", "cheese G", "cheese B", "box mask", "box R", "box G", "box B")
VIDEO = (0, 2, 3, 5, 7, 16)  # the INSTRUCTIONS in the demo video


def match_robots(previous, intended, current):
    """Order current robot positions (N, 2) to follow previous ones: Hungarian matching on the
    distance to where each robot meant to go, plus its distance moved."""
    moved = ((previous[:, None, :] - current[None, :, :]) ** 2).sum(-1)
    off = ((intended[:, None, :] - current[None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(off + 0.1 * moved)
    out = np.zeros_like(current)
    out[rows] = current[cols]
    return out


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


class Recorder:
    """One episode's recording: add(...) once per step, in order; episode() the finished dict."""

    def __init__(self, scene, texts, tokenizer):
        """texts: the commands given in order (one for a single instruction)."""
        self.scene, self.texts, self.tokenizer = scene, texts, tokenizer
        self.positions = np.argwhere(scene["mass"] > 0.5)
        self.codes = None
        self.blobs = []
        self.blob_index = {}
        self.steps = []

    def blob(self, channel):
        key = b64((channel * 255).round().astype(np.uint8))
        if key not in self.blob_index:
            self.blob_index[key] = len(self.blobs)
            self.blobs.append(key)
        return self.blob_index[key]

    def add(self, mass, cheese_pos, cheese_alive, cheese_color, obs, probs, codes, eaten, pushed, command=0, robots=None, now=None):
        """The state at this step (numpy, one scene), the policy's probabilities (H, W, 9) and
        sampled codes (H, W) there, the cheese slots eaten / pushed on the way here, and the index
        of the command in force. MuJoCo physics: robots (N, 2) are the snapped squares by robot id
        and now (N + K, 2) the continuous positions (see add_path)."""
        if robots is not None:
            self.positions = np.asarray(robots)
        elif self.steps:  # follow the robots from the previous step
            intended = self.positions + np.array([DIRS[c] if c != STAY else (0, 0) for c in self.codes])
            self.positions = match_robots(self.positions, intended, np.argwhere(mass > 0.5))
        self.codes = codes[self.positions[:, 0], self.positions[:, 1]]
        expected = probs @ CODE_VECTORS.numpy()  # (H, W, 2): x, y
        self.steps.append(
            {
                "t": len(self.steps),
                "command": command,
                "robots": np.column_stack([self.positions, self.codes]).flatten().tolist(),
                "cheeses": [[int(p[0]), int(p[1]), int(a)] for p, a, c in zip(cheese_pos.tolist(), cheese_alive.tolist(), cheese_color.tolist()) if c >= 0],
                "eaten": [int(k) for k in eaten],
                "pushed": [int(k) for k in pushed],
                "field": b64((expected.transpose(2, 0, 1) * 127).round().astype(np.int8)),
                "pmax": b64((probs.max(-1) * 255).round().astype(np.uint8)),
                "obs": [self.blob(obs[c]) for c in range(len(CHANNELS))],
            }
        )
        if now is not None:
            self.add_path(now[None])

    def add_path(self, path):
        """The continuous positions (frames, N + K, 2) as (row, col), robots by id then cheeses by
        slot, over the step just recorded (frame 0 is its state); float16, base64."""
        self.steps[-1]["path"] = b64(np.asarray(path, np.float16))

    def episode(self, **extra):
        scene = self.scene
        commands = []
        for text in self.texts:
            ids = self.tokenizer(text).input_ids
            commands.append({"text": text, "token_ids": ids, "tokens": [self.tokenizer.decode([i]) for i in ids]})
        eaten_at = next((s["t"] for s in self.steps if s["eaten"]), None)
        return {
            "instruction": " -> ".join(self.texts),
            "commands": commands,
            "robot_count": int(len(self.positions)),
            "walls": b64(scene["walls"].astype(np.uint8)),
            "boxes": [{"rect": list(r), "colour": COLOUR_NAMES[c], "rgb": COLOUR_RGB[c].tolist()} for r, c in zip(scene["boxes"], scene["box_colours"])],
            "cheeses": [{"colour": COLOUR_NAMES[c], "rgb": COLOUR_RGB[c].tolist()} for c in scene["cheese_colours"]],
            "eaten_at": eaten_at,
            "pushes": sum(len(s["pushed"]) for s in self.steps),
            **extra,
            "obs_blobs": self.blobs,
            "steps": self.steps,
        }


def present(scene):
    return {COLOUR_NAMES[c] for c in scene["cheese_colours"]}


def box_rect(scene, colour):
    colours = [COLOUR_NAMES[c] for c in scene["box_colours"]]
    return scene["boxes"][colours.index(colour)] if colour in colours else None


def eat_single(text, a):
    def make(scene):
        if a not in present(scene):
            return None
        return T.Task("logic", text, {"eat": {a}, "any_of": set(), "order": [], "forbid": present(scene) - {a}}, "")

    return make


def eat_either(text, a, b):
    def make(scene):
        if not {a, b} <= present(scene):
            return None
        return T.Task("logic", text, {"eat": set(), "any_of": {a, b}, "order": [], "forbid": present(scene) - {a, b}}, "")

    return make


def eat_all(text):
    def make(scene):
        if not present(scene):
            return None
        return T.Task("logic", text, {"eat": present(scene), "any_of": set(), "order": [], "forbid": set()}, "")

    return make


def eat_nearest_box(text, b):
    """As tasks.logic's nearest_box form: the cheese whose centre is nearest the box's centre."""

    def make(scene):
        rect = box_rect(scene, b)
        if rect is None or not scene["cheeses"]:
            return None
        side = cheese_side(T.SIZE)
        ref = (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
        centres = {COLOUR_NAMES[c]: (y + side / 2, x + side / 2) for (y, x), c in zip(scene["cheeses"], scene["cheese_colours"])}
        target = min(centres, key=lambda c: np.hypot(centres[c][0] - ref[0], centres[c][1] - ref[1]))
        return T.Task("logic", text, {"eat": {target}, "any_of": set(), "order": [], "forbid": present(scene) - {target}}, "")

    return make


def catch(text, c):
    """As tasks.runner's catch form: eat the cheese c, which runs from the swarm (tasks.flee)."""

    def make(scene):
        if c not in present(scene):
            return None
        return T.Task("runner", text, {"eat": {c}, "any_of": set(), "order": [], "forbid": present(scene) - {c}, "form": "single"}, "")

    return make


def form(text, mask):
    """As tasks.formation: fill the mask (scored by F1 of occupied squares against it)."""

    def make(scene):
        if (mask & scene["walls"]).any():
            return None
        return T.Task("formation", text, {"mask": mask}, "")

    return make


def go_box(text, b):
    """As tasks.place's box form: None without the box, or when it cannot hold the swarm (tasks.box_room)."""

    def make(scene):
        rect = box_rect(scene, b)
        if rect is None or not T.box_room(scene, rect):
            return None
        return T.Task("place", text, {"box": rect, "forbid": present(scene)}, "")

    return make


def go_region(text, r):
    return lambda scene: T.Task("place", text, {"region": r, "forbid": present(scene)}, "")


def go_direction(text, d):
    """As tasks.place's move form: None when the swarm has no room to go that way (tasks.move_room)."""

    def make(scene):
        if T.move_room(scene, d) < T.MOVE_NEED + T.MOVE_MARGIN:
            return None
        return T.Task("place", text, {"direction": d, "forbid": present(scene)}, "")

    return make


def scenes_with(cheeses, boxes, cheese_colours=(), box_colours=(), runner="", wall_share=0.5):
    """A scene sampler (tasks.sample_scene: random walls, swarm size and placement) whose scenes
    hold cheeses and boxes of the given colours, and with runner a colour, one runner cheese of
    that colour; None when a draw misses (the caller retries)."""

    def sample(rng):
        scene = T.sample_scene(rng, cheeses=cheeses, boxes=boxes, require=cheese_colours, runners=int(bool(runner)), wall_share=wall_share)
        if scene is None or any(box_rect(scene, b) is None for b in box_colours):
            return None
        if runner and [COLOUR_NAMES[scene["cheese_colours"][k]] for k in scene["runners"]] != [runner]:
            return None
        return scene

    return sample


# Each instruction: its intent keys (tasks.WORDINGS keys, one per command), the commands as
# (text, task maker) in the training wordings, and its scenes (counts as the task makers use).
INSTRUCTIONS = [
    {"keys": ["logic.single"], "commands": [("eat the red cheese", eat_single("eat the red cheese", "red"))], "scenes": scenes_with((1, 6), (0, 2), ("red",))},
    {"keys": ["logic.single"], "commands": [("gobble up the green cheese", eat_single("gobble up the green cheese", "green"))], "scenes": scenes_with((1, 6), (0, 2), ("green",))},
    {"keys": ["logic.or"], "commands": [("eat either yellow or purple", eat_either("eat either yellow or purple", "yellow", "purple"))], "scenes": scenes_with((2, 6), (0, 2), ("yellow", "purple"))},
    {"keys": ["logic.all"], "commands": [("clear the board of cheese", eat_all("clear the board of cheese"))], "scenes": scenes_with((1, 4), (0, 2))},
    {"keys": ["logic.nearest_box"], "commands": [("eat the cheese closest to the white box", eat_nearest_box("eat the cheese closest to the white box", "white"))], "scenes": scenes_with((2, 5), (1, 2), (), ("white",))},
    {"keys": ["place.box"], "commands": [("go into the blue box", go_box("go into the blue box", "blue"))], "scenes": scenes_with((0, 3), (1, 3), (), ("blue",))},
    {"keys": ["place.box"], "commands": [("get inside the orange box", go_box("get inside the orange box", "orange"))], "scenes": scenes_with((0, 3), (1, 3), (), ("orange",))},
    {"keys": ["place.move"], "commands": [("move left", go_direction("move left", "left"))], "scenes": scenes_with((0, 3), (0, 3))},
    {"keys": ["place.move"], "commands": [("head down", go_direction("head down", "down"))], "scenes": scenes_with((0, 3), (0, 3))},
    {"keys": ["place.region"], "commands": [("go to the top left", go_region("go to the top left", "top left"))], "scenes": scenes_with((0, 3), (0, 3))},
    {"keys": ["place.region"], "commands": [("gather in the bottom right", go_region("gather in the bottom right", "bottom right"))], "scenes": scenes_with((0, 3), (0, 3))},
    {"keys": ["formation.shape"], "commands": [("form a ring", form("form a ring", shape_mask("ring", T.SIZE)))], "scenes": scenes_with((0, 2), (0, 1), wall_share=0.0)},
    {"keys": ["formation.shape"], "commands": [("make a plus sign", form("make a plus sign", shape_mask("plus", T.SIZE)))], "scenes": scenes_with((0, 2), (0, 1), wall_share=0.0)},
    {"keys": ["formation.letter"], "commands": [("form the letter L", form("form the letter L", text_mask("L", T.SIZE)))], "scenes": scenes_with((0, 2), (0, 1), wall_share=0.0)},
    {"keys": ["runner.catch"], "commands": [("catch the red cheese", catch("catch the red cheese", "red"))], "scenes": scenes_with((1, 3), (0, 1), runner="red")},
    {
        "keys": ["place.box", "logic.all", "place.move"],
        "commands": [("go into the green box", go_box("go into the green box", "green")), ("eat everything", eat_all("eat everything")), ("move up", go_direction("move up", "up"))],
        "scenes": scenes_with((1, 3), (1, 2), (), ("green",)),
    },
    {  # real-time steering: each command is replaced mid-motion, once it takes effect (Rollout.switch)
        "keys": ["place.box", "logic.single", "place.move"],
        "commands": [("go into the blue box", go_box("go into the blue box", "blue")), ("eat the red cheese", eat_single("eat the red cheese", "red")), ("move left", go_direction("move left", "left"))],
        "scenes": scenes_with((1, 3), (1, 3), ("red",), ("blue",)),
        "switch": (0.5, 80, 15),  # switch once the command has visibly taken effect and run 15 steps; 80 steps only as a safety net
    },
]


def snapshot(world, i):
    return {"mass": world.mass[i].float().cpu().numpy(), "cheese_pos": world.cheese_pos[i].cpu().numpy(), "cheese_alive": world.cheese_alive[i].cpu().numpy()}


class Rollout:
    """One episode's progress through its commands: the command in force, its task (built from
    the scene as it stands when the command is given), when it was met and when it ends."""

    def __init__(self, commands, scene, first, cap, hold, final_hold, switch=None):
        """switch: (score, steps, dwell) for steering: every command but the last is replaced by the
        next once its score reaches score (it has visibly taken effect) and it has run at least dwell
        steps, or after steps; None: a command runs until met plus hold steps."""
        self.commands, self.cap, self.hold, self.final_hold, self.switch = commands, cap, hold, final_hold, switch
        self.segments = []
        self.finished_at = None
        self.begin(0, 0, scene, first)

    def begin(self, j, t, scene, first):
        self.j, self.start, self.scene, self.first = j, t, scene, first
        make = self.commands[j][1]
        self.scored = make is not None  # free text has no checker: it runs cap steps
        self.task = make(scene) if self.scored else None
        self.eaten = []
        self.met_at = None
        self.end = t + self.cap
        if self.scored and self.task is None:  # the scene no longer allows the command (say, its cheese is gone)
            self.segments.append({"text": self.commands[j][0], "start": t, "met_at": None, "end": t, "success": False, "final_score": 0.0, "unasked": False, "eaten": [], "note": "not applicable"})
            self.finished_at = t

    def after_step(self, t, world, i, eaten_slots, runners):
        """Update after the step into state t (t >= 1); returns True when the episode ends at t."""
        for slot in eaten_slots:
            self.eaten.append(COLOUR_NAMES[int(world.cheese_color[i, slot])])
        if not self.scored:
            if t < self.end:
                return False
            return self.close(t, world, i, runners, success=None, score=None, unasked=None)
        score, unasked = T.score(self.task, self.scene, [self.first, snapshot(world, i)], self.eaten)
        last = self.j == len(self.commands) - 1
        steering = self.switch is not None and not last
        if steering:
            took_effect = score >= self.switch[0] and not unasked
            if took_effect and self.met_at is None:
                self.met_at = t
            if not took_effect and t - self.start < self.switch[1]:
                return False
            if took_effect and t - self.start < self.switch[2]:  # let the new behaviour play out before the next command
                return False
            success = took_effect
        else:
            if self.met_at is None and score >= 0.99 and not unasked:
                self.met_at = t
                self.end = min(t + (self.final_hold if last else self.hold), self.start + self.cap)
            if t < self.end:
                return False
            success = self.met_at is not None and score >= 0.99 and not unasked  # met, and still met at the end
        return self.close(t, world, i, runners, bool(success), round(float(score), 3), bool(unasked))

    def close(self, t, world, i, runners, success, score, unasked):
        """End the command in force at t; start the next one. True when the episode ends."""
        self.segments.append({"text": self.commands[self.j][0], "start": self.start, "met_at": self.met_at, "end": t, "success": success, "final_score": score, "unasked": unasked, "eaten": list(self.eaten)})
        if self.j == len(self.commands) - 1:
            self.finished_at = t
            return True
        self.begin(self.j + 1, t, scene_of(world, runners, i), snapshot(world, i))
        return self.finished_at is not None


def lm_act(sample, tokenizer, device):
    """act(world, rollouts) -> (obs, field, probs) from the trained policy."""

    def act(world, rollouts):
        obs = observe(world)
        cond = tokens(tokenizer, [r.commands[r.j][0] for r in rollouts], device)
        _, field, probs = sample(obs, cond)
        return obs, field, probs

    return act


def mujoco_step(batch, world, codes, active):
    """One policy step in MuJoCo: run every active scene, snap it to the grid, eat by the grid's
    rule. Returns (snapped world, eaten (B, K), pushed (B, K), paths per scene)."""
    paths = batch.step(codes, active)
    alive = world.cheese_alive.cpu()
    before = world.cheese_pos.cpu()
    snapped = batch.world(alive)
    eaten = eat_cheeses(snapped)
    batch.remove(eaten.numpy())
    snapped = replace(snapped, cheese_alive=alive & ~eaten)
    pushed = alive & (snapped.cheese_pos != before).any(-1)
    return snapped, eaten, pushed, paths


@torch.no_grad()
def run_batch(act, tokenizer, instruction, scenes, seeds, args, log, on_step=None):
    """Roll out one instruction (its commands in turn) on each scene in one batch. A command runs
    until met plus args.hold steps (args.final_hold for the last), or args.cap steps; with the
    instruction's "switch", every command but the last is replaced as soon as it takes effect
    (Rollout). args.physics: the grid's step, or MuJoCo (mujoco_world) with the grid only what
    the policy sees. Returns the episodes, each with its segments and success (every command met
    and still met at its end). on_step(t), if given, is called after every step."""
    commands = instruction["commands"]
    device = torch.device(args.device)
    world, runners = T.stack(scenes)
    batch = None
    if args.physics == "mujoco":
        assert not runners.any(), "MuJoCo physics has no fleeing cheeses"
        batch = Batch(scenes, world)
    world = world.to(device)
    rollouts = [Rollout(commands, scene, snapshot(world, i), args.cap, args.hold, args.final_hold, instruction.get("switch")) for i, scene in enumerate(scenes)]
    recorders = [Recorder(scene, [text for text, _ in commands], tokenizer) for scene in scenes]
    eaten = torch.zeros(world.cheese_alive.shape, dtype=torch.bool)
    pushed = torch.zeros(world.cheese_alive.shape, dtype=torch.bool)
    t = 0
    while True:
        obs, field, probs = act(world, rollouts)
        codes = encode_field(field).cpu().numpy()
        obs_np, probs_np = obs.cpu().numpy(), probs.cpu().numpy()
        for i, (r, recorder) in enumerate(zip(rollouts, recorders)):
            if r.finished_at is None or t == r.finished_at:
                robots = now = None
                if batch is not None:
                    robots, now = batch.squares[i], batch.swarms[i].positions()
                recorder.add(world.mass[i].cpu().numpy(), world.cheese_pos[i].cpu(), world.cheese_alive[i].cpu(), world.cheese_color[i].cpu(), obs_np[i], probs_np[i], codes[i], eaten[i].nonzero().flatten().tolist(), pushed[i].nonzero().flatten().tolist(), r.j, robots, now)
        if all(r.finished_at is not None for r in rollouts):
            break
        if batch is None:
            world, info = step(world, field)
            eaten, pushed = info.eaten.cpu(), info.pushed.cpu()
        else:
            active = [r.finished_at is None for r in rollouts]
            world, eaten, pushed, paths = mujoco_step(batch, world, codes, active)
            world = world.to(device)
            for recorder, path in zip(recorders, paths):
                if path is not None:
                    recorder.add_path(path)
        t += 1
        if on_step is not None:
            on_step(t)
        if runners.any() and t % T.RUNNER_PERIOD == 0:  # runner cheeses flee, as in tasks.run
            world = replace(world, cheese_pos=T.flee(world, runners))
        for i, r in enumerate(rollouts):
            if r.finished_at is None and r.after_step(t, world, i, eaten[i].nonzero().flatten().tolist(), runners):
                log(outcome(instruction, seeds[i], r, recorders[i]))
    return [recorder.episode(**outcome(instruction, seeds[i], r, recorder)) for i, (r, recorder) in enumerate(zip(rollouts, recorders))]


def outcome(instruction, seed, rollout, recorder):
    return {
        "instruction": " -> ".join(text for text, _ in instruction["commands"]),
        "intents": instruction["keys"],
        "seed": seed,
        "success": len(rollout.segments) == len(instruction["commands"]) and all(s["success"] is True for s in rollout.segments),
        "final_success": len(rollout.segments) == len(instruction["commands"]) and rollout.segments[-1]["success"] is True,  # the last command met
        "length": rollout.finished_at,  # steps
        "pushes": sum(len(s["pushed"]) for s in recorder.steps),
        "segments": rollout.segments,
    }


def rank(episode):
    """Order of preference among an instruction's runs: successes first, then runs that push a
    cheese, then the shortest."""
    return (episode["success"], episode["pushes"] > 0, -episode["length"])


def record_suite(args, log, save):
    """Roll out every instruction (args.only: a subset); save(results) after each, so a stopped run
    keeps what it finished."""
    device = torch.device(args.device)
    policy, tokenizer = load_policy(args.policy, device, args.checkpoint)
    act = lm_act(lm_probs_sampler(policy, args.guidance, T.SIZE // PATCH), tokenizer, device)
    results = []
    for k, instruction in enumerate(INSTRUCTIONS):
        if args.only and str(k) not in args.only.split(","):
            continue
        scenes, seeds = [], []
        seed = 0
        while len(scenes) < args.rollouts:
            scene = instruction["scenes"](np.random.default_rng([k, seed]))
            if scene is not None and all(make(scene) is not None for _, make in instruction["commands"][:1]):
                scenes.append(scene)
                seeds.append(seed)
            seed += 1
        torch.manual_seed(k)
        episodes = run_batch(act, tokenizer, instruction, scenes, seeds, args, log)
        episodes.sort(key=rank, reverse=True)
        wins = sum(e["success"] for e in episodes)
        finals = sum(e["final_success"] for e in episodes)
        text = episodes[0]["instruction"]
        log({"instruction": text, "intents": instruction["keys"], "physics": args.physics, "success_rate": f"{wins}/{len(episodes)}", "final_command_rate": f"{finals}/{len(episodes)}"})
        results.append(
            {
                "instruction": text,
                "intents": instruction["keys"],
                "steering": "switch" in instruction,
                "successes": wins,
                "final_successes": finals,
                "rollouts": len(episodes),
                "outcomes": [{key: e[key] for key in ("seed", "success", "final_success", "length", "pushes", "segments")} for e in episodes],
                "runs": episodes[: args.keep],  # the best runs, best first
            }
        )
        save(results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=HUB_REPO, help="a Hugging Face repo or a local directory with config.json and the checkpoint")
    parser.add_argument("--checkpoint", default=WEIGHTS)
    parser.add_argument("--physics", choices=("grid", "mujoco"), default="mujoco", help="grid: world.step; mujoco: MuJoCo is the world, snapped to the grid for the policy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--rollouts", type=int, default=8, help="rollouts (fresh scenes) per instruction, one batch")
    parser.add_argument("--cap", type=int, default=140, help="steps a command may take at most")
    parser.add_argument("--hold", type=int, default=10, help="steps a command keeps going once met before the next command")
    parser.add_argument("--final_hold", type=int, default=15, help="the same for the last command")
    parser.add_argument("--keep", type=int, default=6, help="best runs kept per instruction")
    parser.add_argument("--only", default=",".join(map(str, VIDEO)), help="comma-separated indices into INSTRUCTIONS (empty: all)")
    parser.add_argument("--out", default="runs/demo/recordings.json")
    parser.add_argument("--log", default="")
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.log or out_path.with_suffix(".log"), "w")

    def log(row):
        line = json.dumps(row)
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log({"pid": os.getpid(), "physics": args.physics, "started": time.strftime("%Y-%m-%d %H:%M:%S")})

    def save(instructions):
        out_path.write_text(json.dumps(recordings(instructions, args), separators=(",", ":")))
        log({"wrote": str(out_path), "mb": round(out_path.stat().st_size / 1e6, 2)})

    save(record_suite(args, log, save))


def recordings(instructions, args):
    """The recordings file: the instructions' runs and what a reader needs to decode them."""
    return {
        "policy": args.policy,
        "guidance": args.guidance,
        "physics": args.physics,
        "frames_per_step": FRAMES_PER_STEP if args.physics == "mujoco" else None,  # rows of each step's "path"
        "size": T.SIZE,
        "cheese_side": cheese_side(T.SIZE),
        "codes": "0-7: world.DIRS (dy, dx) = " + json.dumps(DIRS) + "; 8: stay",
        "code_vectors": CODE_VECTORS.tolist(),
        "channels": CHANNELS,
        "instructions": instructions,
    }


if __name__ == "__main__":
    main()
