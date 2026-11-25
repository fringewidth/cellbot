"""The live demo's server: every visitor gets their own swarm in MuJoCo and steers it by typing.

Each step, one batched call to the policy gives every active swarm its moves; MuJoCo then runs
one second of physics per swarm and snaps it back to the grid the policy reads. The browser
(site/index.html) draws the continuous positions with three.js. The policy call is a function, so
the same server runs locally on your own GPU or, on Modal (deploy/modal_app.py), with the policy
on a GPU elsewhere and a budget that stops it.

    uv run python -m cellbot.demo.server            # then open http://127.0.0.1:8790

API (JSON): POST /api/new {sid?} -> the scene; POST /api/command {sid, text};
GET /api/frame?sid=..&after=t -> the steps after t; GET /api/status.
"""

import argparse
import base64
import secrets
import threading
import time
from pathlib import Path

import numpy as np
import torch

from cellbot import steer
from cellbot.demo.record import mujoco_step
from cellbot.policy.actions import encode_field
from cellbot.sim.mujoco_world import FRAMES_PER_STEP, Batch
from cellbot.sim.world import COLOUR_NAMES, COLOUR_RGB, cheese_side, observe
from cellbot.tasks import suite as T

SITE = Path(__file__).parents[3] / "site" / "index.html"
SUGGESTIONS = {"place.region": 0.5, "place.box": 0.5, "logic.single": 0.7, "logic.all": 0.6, "logic.or": 0.5}  # intents the policy follows reliably
KEEP_STEPS = 40  # steps a session keeps for the browser to fetch


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


class Session:
    """One visitor's swarm: the grid World the policy reads and its MuJoCo twin."""

    def __init__(self, scene):
        self.scene = scene
        world, _ = T.stack([scene])
        self.world = world
        self.batch = Batch([scene], world)
        self.text = ""
        self.t = 0
        self.since = 0  # the step the current command was given at
        self.seen = time.time()
        self.steps = []  # recent steps for the browser: {t, path, alive, events}
        self.suggestion = ""

    def start(self):
        """What the browser needs to draw the scene."""
        world = self.world
        boxes = [[*r, c] for r, c in zip(world.box_rect[0].tolist(), world.box_color[0].tolist()) if c >= 0]
        colours = [c for c in world.cheese_color[0].tolist() if c >= 0]
        return {
            "size": T.SIZE,
            "cheese_side": cheese_side(T.SIZE),
            "walls": "".join("1" if v else "0" for v in world.walls[0].flatten().tolist()),
            "boxes": boxes,
            "cheese_colours": colours,
            "robots": int(self.batch.swarms[0].n_robots),
            "positions": b64(self.batch.swarms[0].positions().astype(np.float16)),
            "t": self.t,
        }

    def frame(self, after):
        return {
            "t": self.t,
            "steps": [s for s in self.steps if s["t"] > after],
            "command": self.text,
            "suggestion": self.suggestion,
        }


class Server:
    """Sessions and the stepping loop. policy(obs (B, C, G, G) float32, texts) -> codes (B, G, G)
    uint8. budget, if given, has allow() -> str ("" when steering may go on, else why not) and
    charge(seconds) for the time the policy's GPU was busy.

    Each visitor (sid) gets `prompts` instructions in all, across resets, and each instruction runs at
    most `command_steps` steps before the swarm rests, so an open tab does not keep the GPU busy."""

    def __init__(self, policy, budget=None, max_sessions=6, prompts=10, command_steps=150, steps_per_second=4.0):
        self.policy, self.budget = policy, budget
        self.max_sessions, self.prompts, self.command_steps, self.period = max_sessions, prompts, command_steps, 1.0 / steps_per_second
        self.used = {}  # sid -> instructions given
        self.sessions = {}
        self.lock = threading.Lock()
        self.rng = np.random.default_rng()
        self.halted = ""  # why steering is stopped for everyone (the budget), or ""
        threading.Thread(target=self.loop, daemon=True).start()

    def new(self, sid):
        with self.lock:
            self.evict()
            if sid not in self.sessions and len(self.sessions) >= self.max_sessions:
                return None
            scene = None
            while scene is None:
                scene = T.sample_scene(self.rng, cheeses=(2, 5), boxes=(0, 2))
            session = Session(scene)
            session.suggestion = steer.suggest(scene, SUGGESTIONS, self.rng) or ""
            self.sessions[sid] = session
            return session.start()

    def command(self, sid, text):
        with self.lock:
            session = self.sessions.get(sid)
            if session is None:
                return None
            session.seen = time.time()
            halted = self.budget.allow() if self.budget is not None else ""
            if halted:
                self.halted = halted
                return {"error": halted}
            if self.used.get(sid, 0) >= self.prompts:
                return {"error": f"You have used all {self.prompts} prompts of this session. Thanks for trying it!"}
            self.used[sid] = self.used.get(sid, 0) + 1
            session.text = text
            session.since = session.t
            return {"ok": True, "prompts_left": self.prompts - self.used[sid]}

    def frame(self, sid, after):
        with self.lock:
            session = self.sessions.get(sid)
            if session is None:
                return None
            session.seen = time.time()
            return {**session.frame(after), "halted": self.halted, "prompts_left": self.prompts - self.used.get(sid, 0)}

    def evict(self, idle=120):
        now = time.time()
        for sid in [k for k, s in self.sessions.items() if now - s.seen > idle]:
            self.sessions.pop(sid)

    def loop(self):
        while True:
            t0 = time.time()
            try:
                self.tick()
            except Exception as error:  # keep serving; the next tick tries again
                print("tick failed:", repr(error), flush=True)
            time.sleep(max(0.0, self.period - (time.time() - t0)))

    def tick(self):
        with self.lock:
            now = time.time()
            for s in self.sessions.values():  # an instruction runs command_steps steps, then the swarm rests
                if s.text and s.t - s.since >= self.command_steps:
                    s.text = ""
            active = [(sid, s) for sid, s in self.sessions.items() if s.text and now - s.seen < 15]
            if not active:
                return
            if self.budget is not None:
                self.halted = self.budget.allow()
                if self.halted:
                    for _, s in active:
                        s.text = ""
                    return
            obs = observe(steer.concat([s.world for _, s in active]))
            texts = [s.text for _, s in active]
        started = time.time()
        codes = self.policy(obs.numpy().astype(np.float32), texts)
        if self.budget is not None:
            self.budget.charge(time.time() - started)
        with self.lock:
            for (sid, s), c in zip(active, codes):
                if self.sessions.get(sid) is not s or not s.text:
                    continue
                self.advance(s, np.asarray(c)[None])

    def advance(self, s, codes):
        alive_before = s.world.cheese_alive.clone()
        world, eaten, pushed, paths = mujoco_step(s.batch, s.world, codes, [True])
        s.world = world
        s.t += 1
        events = [f"ate {COLOUR_NAMES[int(world.cheese_color[0, k])]}" for k in eaten[0].nonzero().flatten().tolist() if alive_before[0, k]]
        if events:
            s.suggestion = steer.suggest(steer.scene_of(world, torch.zeros(1, world.cheese_alive.shape[1], dtype=torch.bool)), SUGGESTIONS, self.rng, avoid=(s.text,)) or ""
        s.steps.append({"t": s.t, "path": b64(paths[0].astype(np.float16)), "alive": world.cheese_alive[0].tolist(), "events": events})
        del s.steps[:-KEEP_STEPS]


def local_policy(source, device, guidance=2.0):
    """policy(obs, texts) -> codes, from the weights on this machine's device."""
    from cellbot.policy.inference import lm_probs_sampler, load_policy, tokens
    from cellbot.policy.model import PATCH

    model, tokenizer = load_policy(source, device)
    sample = lm_probs_sampler(model, guidance, T.SIZE // PATCH)

    @torch.no_grad()
    def policy(obs, texts):
        _, field, _ = sample(torch.from_numpy(obs).to(device), tokens(tokenizer, texts, device))
        return encode_field(field).cpu().numpy().astype(np.uint8)

    return policy


def app(server, page=SITE):
    """The FastAPI app: the page at / and the JSON API."""
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse

    api = FastAPI()
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    palette = {name: [round(float(v), 3) for v in rgb] for name, rgb in zip(COLOUR_NAMES, COLOUR_RGB.tolist())}

    @api.get("/")
    def index():
        return FileResponse(page)

    @api.get("/replay.json")
    def recorded():
        return FileResponse(page.with_name("replay.json"))

    @api.get("/api/status")
    def status():
        return {"halted": server.halted, "sessions": len(server.sessions), "max_sessions": server.max_sessions, "colours": COLOUR_NAMES, "rgb": palette, "frames_per_step": FRAMES_PER_STEP, "step_seconds": server.period}

    @api.post("/api/new")
    async def new(request: Request):
        payload = await request.json()
        sid = str(payload.get("sid") or secrets.token_hex(8))[:32]
        body = server.new(sid)
        if body is None:
            return JSONResponse({"error": "the demo is full right now; try again in a minute"}, 503)
        return {"sid": sid, **body}

    @api.post("/api/command")
    async def command(request: Request):
        payload = await request.json()
        text = " ".join(str(payload.get("text", "")).split())[:120]
        body = server.command(str(payload.get("sid", "")), text) if text else None
        if body is None:
            return JSONResponse({"error": "unknown session"}, 404)
        return JSONResponse(body, 429 if "error" in body else 200)

    @api.get("/api/frame")
    def frame(sid: str, after: int = -1):
        body = server.frame(sid, after)
        return body if body is not None else JSONResponse({"error": "unknown session"}, 404)

    return api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="fringewidth/cellbot1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    import uvicorn

    server = Server(local_policy(args.policy, torch.device(args.device)))
    print(f"open http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app(server), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
