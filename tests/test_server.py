"""The live server with a random policy: sessions, stepping in MuJoCo, frames, the budget stop."""

import time

import numpy as np

from cellbot.demo import server as S
from cellbot.tasks import suite as T


def random_policy(obs, texts):
    assert len(obs) == len(texts)
    return np.random.randint(0, 9, (len(obs), T.SIZE, T.SIZE)).astype(np.uint8)


class Spent:
    def allow(self):
        return "out of budget"

    def charge(self, seconds):
        raise AssertionError("no policy call once the budget is spent")


def wait_for(condition, seconds=10):
    end = time.time() + seconds
    while time.time() < end and not condition():
        time.sleep(0.05)
    return condition()


def test_sessions_step_and_report_frames():
    server = S.Server(random_policy, steps_per_second=20)
    start = server.new("a")
    assert len(start["walls"]) == T.SIZE * T.SIZE and start["robots"] > 0
    assert server.frame("a", -1)["steps"] == []  # no command, no steps
    assert server.command("a", "eat the red cheese") == {"ok": True, "prompts_left": 9}
    assert wait_for(lambda: server.frame("a", -1)["t"] >= 3)
    frame = server.frame("a", 1)
    assert all(s["t"] > 1 for s in frame["steps"])
    n = start["robots"] + len(start["cheese_colours"])
    path = np.frombuffer(S.base64.b64decode(frame["steps"][0]["path"]), np.float16)
    assert path.size % (n * 2) == 0
    assert server.frame("nobody", -1) is None


def test_a_spent_budget_stops_steering():
    server = S.Server(random_policy, budget=Spent(), steps_per_second=20)
    server.new("a")
    server.command("a", "move left")
    assert wait_for(lambda: server.halted == "out of budget")
    assert server.frame("a", -1)["t"] == 0
    assert server.command("a", "move right") == {"error": "out of budget"}


def test_prompts_per_session_survive_a_reset():
    server = S.Server(random_policy, prompts=2, steps_per_second=20)
    server.new("a")
    assert "ok" in server.command("a", "move left")
    server.new("a")  # reset: a new scene, the same visitor
    assert "ok" in server.command("a", "move right")
    assert "error" in server.command("a", "stop")


def test_an_instruction_stops_after_command_steps():
    server = S.Server(random_policy, command_steps=3, steps_per_second=40)
    server.new("a")
    server.command("a", "move left")
    assert wait_for(lambda: server.frame("a", -1)["command"] == "")
    assert server.frame("a", -1)["t"] == 3
