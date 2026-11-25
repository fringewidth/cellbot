"""The live demo on Modal: the site and the MuJoCo physics on a small CPU container, the policy on a
T4 that runs only while someone is steering. Both scale to zero.

    uvx modal deploy deploy/modal_app.py

Spending is capped in code (Budget): the month's real Modal spend, read from the billing API, must
stay under 90% of the free monthly credit times the share of the month gone by; past that the site
says "check back tomorrow". Each visitor gets 10 prompts, and each prompt runs at most 150 steps.
Set a Modal spend limit of $0 as well, so the free monthly credit is the most it can ever cost.
"""

import calendar
import time
from datetime import date
from pathlib import Path

import modal

ROOT = Path(__file__).parents[1]
MONTHLY_CREDIT = 30.0  # dollars of free compute per month on Modal's Starter plan
SAFE_SHARE = 0.9  # of the credit the demo may spend
GPU_DOLLARS_PER_SECOND = 0.000164 + 2 * 0.0000131 + 8 * 0.00000222  # T4, 2 cores, 8 GiB: an estimate between billing reads
COLD_START_SECONDS = 40  # added each time the GPU wakes up
IDLE_SECONDS = 60  # the GPU's scaledown window, added each time it goes idle
BILLING_LAG_SECONDS = 2 * 3600  # estimates this recent are counted on top of the billed cost
LIMIT_MESSAGE = "Daily usage limit reached. Check back tomorrow!"

app = modal.App("cellbot")
usage = modal.Dict.from_name("cellbot-usage", create_if_missing=True)


def download():
    from huggingface_hub import snapshot_download

    snapshot_download("Qwen/Qwen3-0.6B")
    snapshot_download("fringewidth/cellbot1")


gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=4.56", "safetensors", "huggingface_hub", "numpy", "scipy")
    .run_function(download)
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir(ROOT / "src" / "cellbot", "/root/cellbot")
)
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("mujoco", "numpy", "scipy", "fastapi", "uvicorn")
    .add_local_dir(ROOT / "src" / "cellbot", "/root/cellbot")
    .add_local_dir(ROOT / "site", "/root/site")
)


@app.cls(image=gpu_image, gpu="T4", max_containers=1, scaledown_window=IDLE_SECONDS, timeout=600)
class Policy:
    @modal.enter()
    def load(self):
        import torch

        from cellbot.policy.inference import lm_probs_sampler, load_policy
        from cellbot.policy.model import PATCH

        self.device = torch.device("cuda")
        self.model, self.tokenizer = load_policy("fringewidth/cellbot1", self.device)
        self.sample = lm_probs_sampler(self.model, 2.0, 32 // PATCH)

    @modal.method()
    def moves(self, obs: bytes, shape: tuple, texts: list) -> bytes:
        """obs: float16 (B, C, G, G) bytes; returns uint8 codes (B, G, G) bytes."""
        import numpy as np
        import torch

        from cellbot.policy.actions import encode_field
        from cellbot.policy.inference import tokens

        x = torch.from_numpy(np.frombuffer(obs, np.float16).reshape(shape).astype(np.float32)).to(self.device)
        with torch.no_grad():
            _, field, _ = self.sample(x, tokens(self.tokenizer, texts, self.device))
        return encode_field(field).cpu().numpy().astype(np.uint8).tobytes()


class Budget:
    """Steering is allowed while this month's Modal spend stays under a line that rises through the
    month: SAFE_SHARE of the monthly credit times the share of the month gone by. That makes the limit
    daily, so "check back tomorrow" holds. Spend is read from Modal's billing API every few minutes;
    GPU time since then is added as an estimate, and recent estimates are kept for a while because
    billing reports lag. If billing cannot be read, the estimates alone count (kept in a modal.Dict)."""

    REFRESH_SECONDS = 300

    def __init__(self):
        self.billed = None
        self.read_at = 0.0
        self.recent = []  # (time, dollars) estimated
        self.last_call = 0.0
        self.unflushed = 0.0

    def line(self):
        today = date.today()
        return MONTHLY_CREDIT * SAFE_SHARE * today.day / calendar.monthrange(today.year, today.month)[1]

    def spent(self):
        now = time.time()
        month = "estimate-" + date.today().isoformat()[:7]
        if now - self.read_at > self.REFRESH_SECONDS:
            self.read_at = now
            try:
                self.billed = float(modal.Workspace.from_context().billing.summary("this month").metered_cost)
            except Exception as error:  # e.g. the container may not read billing: fall back to estimates
                print("billing not readable:", repr(error), flush=True)
                self.billed = None
            if self.unflushed:
                usage[month] = usage.get(month, 0.0) + self.unflushed
                self.unflushed = 0.0
        self.recent = [(t, d) for t, d in self.recent if now - t < BILLING_LAG_SECONDS]
        if self.billed is None:
            return usage.get(month, 0.0) + self.unflushed
        return self.billed + sum(d for _, d in self.recent)

    def allow(self):
        return LIMIT_MESSAGE if self.spent() >= self.line() else ""

    def charge(self, seconds):
        now = time.time()
        if now - self.last_call > IDLE_SECONDS:  # the GPU had shut down: a cold start, and the idle minute after
            seconds += COLD_START_SECONDS + IDLE_SECONDS
        else:
            seconds = max(seconds, now - self.last_call)  # the GPU stayed up in between
        self.last_call = now
        dollars = seconds * GPU_DOLLARS_PER_SECOND
        self.recent.append((now, dollars))
        self.unflushed += dollars


@app.function(image=web_image, cpu=(0.25, 2.0), memory=(1024, 3072), max_containers=1, scaledown_window=120, timeout=24 * 3600)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="cellbot")
def web():
    import numpy as np

    from cellbot.demo import server as S

    policy = Policy()

    def remote_policy(obs, texts):
        data = policy.moves.remote(obs.astype(np.float16).tobytes(), obs.shape, texts)
        return np.frombuffer(data, np.uint8).reshape(obs.shape[0], obs.shape[2], obs.shape[3])

    server = S.Server(remote_policy, budget=Budget(), max_sessions=6, prompts=10, command_steps=150)
    return S.app(server, page=Path("/root/site/index.html"))
