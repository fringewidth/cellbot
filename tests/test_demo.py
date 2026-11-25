"""The demo pipeline end to end with a random policy: a MuJoCo rollout of free-text commands, then
the video (needs ffmpeg)."""

import shutil
from types import SimpleNamespace

import pytest
import torch

from cellbot.demo import live, record, render
from cellbot.policy.actions import decode_field
from cellbot.sim.world import observe
from cellbot.tasks import suite as T


def random_act(world, rollouts):
    probs = torch.softmax(torch.randn(len(rollouts), T.SIZE, T.SIZE, 9), -1)
    codes = torch.distributions.Categorical(probs=probs).sample()
    return observe(world), decode_field(codes), probs


class CharTokenizer:
    """Stands in for the Qwen tokenizer: the recorder only stores token ids and pieces."""

    def __call__(self, text):
        return SimpleNamespace(input_ids=[ord(c) for c in text])

    def decode(self, ids):
        return "".join(map(chr, ids))


def test_free_text_runs_each_command_for_its_steps(tmp_path):
    args = SimpleNamespace(device="cpu", physics="mujoco", cap=3, hold=10, final_hold=15)
    instruction = {"keys": [], "commands": [("go left", None), ("go right", None)]}
    steps = []
    (episode,) = record.run_batch(random_act, CharTokenizer(), instruction, [live.free_scene(0)], [0], args, lambda row: None, steps.append)
    assert [s["start"] for s in episode["segments"]] == [0, 3]
    assert [s["success"] for s in episode["segments"]] == [None, None]
    assert len(episode["steps"]) == 7 and steps == list(range(1, 7))
    assert "path" in episode["steps"][0]
    if shutil.which("ffmpeg") is None:
        pytest.skip("no ffmpeg")
    episode["steering"] = True
    live.save_video(episode, tmp_path / "run.mp4", height=360)
    assert (tmp_path / "run.mp4").stat().st_size > 10_000


def test_video_tasks_have_scenes():
    for name, (k, seed) in live.VIDEO_TASKS.items():
        assert record.INSTRUCTIONS[k]["commands"][0][0] == name.split(" -> ")[0]
        assert live.task_scene(k, seed) is not None


def test_muted_keeps_grey_and_cuts_saturation():
    assert render.muted([0.5, 0.5, 0.5]).tolist() == pytest.approx([0.5, 0.5, 0.5])
    red = render.muted([0.9, 0.24, 0.24])
    assert red[0] > red[1] == pytest.approx(red[2]) and red[0] - red[1] < 0.3 * 0.66 + 1e-6
