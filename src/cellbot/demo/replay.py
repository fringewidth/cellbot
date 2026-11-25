"""A recorded MuJoCo run as the website's replay (site/replay.json): what the page plays while the
live model is asleep.

    uv run python -m cellbot.demo.replay runs/demo/steer-mujoco.json --out site/replay.json
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from cellbot.sim.world import COLOUR_NAMES


def replay(episode, size=32, side=4):
    """The page's format: start (as the server's /api/new) and steps (as /api/frame), with the
    index of the command in force at each step."""
    n, k = episode["robot_count"], len(episode["cheeses"])
    walls = np.frombuffer(base64.b64decode(episode["walls"]), np.uint8).reshape(size, size)
    first = np.frombuffer(base64.b64decode(episode["steps"][0]["path"]), np.float16).reshape(-1, n + k, 2)[0]
    start = {
        "size": size,
        "cheese_side": side,
        "walls": "".join("1" if v else "0" for v in walls.flatten().tolist()),
        "boxes": [[*b["rect"], COLOUR_NAMES.index(b["colour"])] for b in episode["boxes"]],
        "cheese_colours": [COLOUR_NAMES.index(c["colour"]) for c in episode["cheeses"]],
        "robots": n,
        "positions": base64.b64encode(first.tobytes()).decode(),
    }
    steps = []
    for step in episode["steps"]:
        if "path" not in step:
            continue
        events = [f"ate {episode['cheeses'][s]['colour']}" for s in step["eaten"]]
        steps.append({"t": step["t"], "path": step["path"], "alive": [bool(c[2]) for c in step["cheeses"]], "events": events, "command": step.get("command", 0)})
    return {"commands": [c["text"] for c in episode["commands"]], "start": start, "steps": steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", help="cellbot.demo.record --physics mujoco output")
    parser.add_argument("--instruction", type=int, default=0, help="index into the recordings' instructions")
    parser.add_argument("--run", type=int, default=0, help="index into its runs (0: the best)")
    parser.add_argument("--out", default="site/replay.json")
    args = parser.parse_args()
    data = json.loads(Path(args.recordings).read_text())
    episode = data["instructions"][args.instruction]["runs"][args.run]
    assert episode["success"], "the replay must show a run that succeeds"
    Path(args.out).write_text(json.dumps(replay(episode, data["size"], data["cheese_side"]), separators=(",", ":")))
    print("wrote", args.out, round(Path(args.out).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
