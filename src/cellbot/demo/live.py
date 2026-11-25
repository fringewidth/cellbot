"""One run of cellbot1 on demand, rendered as in the demo video: the Hugging Face Space and the
command line both use this.

    uv run python -m cellbot.demo.live "eat the red cheese" --out runs/demo/live.mp4
    uv run python -m cellbot.demo.live "go into the blue box" "move left" --steps 60 --out runs/demo/live.mp4
    uv run python -m cellbot.demo.live --task 0 --out runs/demo/live.mp4    # a task from the video, with its checker

A task from the video (record.INSTRUCTIONS) is scored by its checker and ends once met. Free text
has no checker: each command runs a fixed number of steps, then the next one takes over.
"""

import argparse
from types import SimpleNamespace

import numpy as np
import torch

from cellbot.demo import record, render
from cellbot.policy.inference import HUB_REPO, WEIGHTS, lm_probs_sampler, load_policy
from cellbot.policy.model import PATCH
from cellbot.sim.world import cheese_side
from cellbot.tasks import suite as T

# The video's tasks: (index into record.INSTRUCTIONS, the scene seed of the run shown in the video).
VIDEO_TASKS = {
    "eat the red cheese": (0, 8),
    "eat either yellow or purple": (2, 0),
    "clear the board of cheese": (3, 2),
    "go into the blue box": (5, 11),
    "move left": (7, 1),
    "go into the blue box -> eat the red cheese -> move left": (16, 9),
}


def task_scene(k, seed):
    """The scene of record.INSTRUCTIONS[k] at this seed, drawn as record.record_suite draws it."""
    instruction = record.INSTRUCTIONS[k]
    while True:
        scene = instruction["scenes"](np.random.default_rng([k, seed]))
        if scene is not None and instruction["commands"][0][1](scene) is not None:
            return scene
        seed += 1


def free_scene(seed):
    """A scene for free text: 1-4 cheeses, up to 2 boxes, random walls."""
    rng = np.random.default_rng(seed)
    while True:
        scene = T.sample_scene(rng, cheeses=(1, 4), boxes=(0, 2))
        if scene is not None:
            return scene


class Runner:
    """The policy loaded once; run(...) rolls out one scene in MuJoCo and returns its episode."""

    def __init__(self, source=HUB_REPO, checkpoint=WEIGHTS, device=None, guidance=2.0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.policy, self.tokenizer = load_policy(source, self.device, checkpoint)
        self.act = record.lm_act(lm_probs_sampler(self.policy, guidance, T.SIZE // PATCH), self.tokenizer, self.device)

    def run(self, instruction, scene, cap, on_step=None):
        """instruction: a record.INSTRUCTIONS entry, or {"commands": [(text, None), ...]} for free text."""
        args = SimpleNamespace(device=str(self.device), physics="mujoco", cap=cap, hold=10, final_hold=15)
        instruction = {"keys": [], **instruction}
        (episode,) = record.run_batch(self.act, self.tokenizer, instruction, [scene], [0], args, lambda row: None, on_step)
        episode["steering"] = len(instruction["commands"]) > 1
        return episode

    def task(self, name, seed=None, on_step=None):
        """A task from the video, on its video scene unless seed is given."""
        k, video_seed = VIDEO_TASKS[name]
        return self.run(record.INSTRUCTIONS[k], task_scene(k, video_seed if seed is None else seed), cap=140, on_step=on_step)

    def free(self, commands, seed=0, steps=60, on_step=None):
        """Free-text commands, given one after another, steps each."""
        return self.run({"commands": [(text, None) for text in commands]}, free_scene(seed), cap=steps, on_step=on_step)


def save_video(episode, path, height=720, frames_per_step=3, label="", on_step=None):
    """The episode in the video's look: MuJoCo beside the grid the policy reads."""
    layout = render.Layout(height)
    video = render.Video(path, layout)
    frames = render.STEER_FRAMES if episode["steering"] else frames_per_step
    render.render_task(video, episode, episode, label, frames, layout, T.SIZE, cheese_side(T.SIZE), "Cellular automaton (the grid the policy reads)", on_step)
    video.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("commands", nargs="*", help="free-text commands, given one after another")
    parser.add_argument("--task", type=int, default=-1, help="instead: a task from the video, 0-5")
    parser.add_argument("--seed", type=int, default=None, help="the scene (default: the video's scene for --task, 0 for free text)")
    parser.add_argument("--steps", type=int, default=60, help="free text: steps per command")
    parser.add_argument("--policy", default=HUB_REPO)
    parser.add_argument("--device", default=None)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", default="runs/demo/live.mp4")
    args = parser.parse_args()
    runner = Runner(args.policy, device=args.device)
    if args.task >= 0:
        episode = runner.task(list(VIDEO_TASKS)[args.task], args.seed)
    else:
        episode = runner.free(args.commands, args.seed or 0, args.steps)
    for segment in episode["segments"]:
        print(segment)
    save_video(episode, args.out, args.height)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
