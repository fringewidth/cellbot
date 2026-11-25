"""The demo video: each task shows one scene run in MuJoCo (left) beside the grid the policy
reads, a cellular automaton (right). Grey and white, with the cheeses and boxes in colour.

    uv run python -m cellbot.demo.render --recordings runs/demo/mujoco.json --grid runs/demo/grid.json --out runs/demo/cellbot1.mp4
    uv run python -m cellbot.demo.render --recordings a.json b.json --grid a-grid.json b-grid.json --out all.mp4   # several recordings in one video

With --grid the right panel is the same scene (same seed) rolled out on the raw grid; without it,
the right panel is the MuJoCo run itself as the policy saw it, snapped to the grid each step.
"""

import argparse
import base64
import json
import math
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = Path(__file__).with_name("fonts") / "OpenSans.ttf"
FPS = 30
HOLD_SECONDS = 0.7  # the last frame of a task stays this long
CARD_SECONDS = 2.5
TYPE_SECONDS = 0.5  # a new command is typed in over this long, ending as it takes over
HIGHLIGHT_SECONDS = 1.5  # a panel's command label stays bright this long after it changes
STEER_FRAMES = 5  # frames per policy step in a steering task (slower, so each change can be seen)
GREY = (170, 170, 170)


def font(size, weight=b"Regular"):
    face = ImageFont.truetype(str(FONT), size)
    face.set_variation_by_name(weight)
    return face


def grid(blob, dtype, shape):
    return np.frombuffer(base64.b64decode(blob), dtype).reshape(shape)


def robots_of(step):
    return np.array(step["robots"]).reshape(-1, 3)


def square_xy(row, col, size):
    """World (x, y) of a square's centre: columns run along +x, row 0 is the far side (+y)."""
    return col + 0.5, size - row - 0.5


def rgba(rgb):
    return " ".join(f"{v:.2f}" for v in rgb) + " 1"


class Layout:
    """Pixel positions for a frame height (1080 is the reference; everything scales with it)."""

    def __init__(self, height=1080):
        s = height / 1080
        self.s = s
        self.width, self.height = round(1920 * s) // 2 * 2, round(height) // 2 * 2
        self.columns = (round(40 * s), round(980 * s))  # left: MuJoCo, right: the grid
        self.view = (round(900 * s), round(860 * s))  # each world view: width, height
        self.top, self.top_steer = round(150 * s), round(190 * s)

    def px(self, v):
        return max(1, round(v * self.s))


# ---------------------------------------------------------------------------------------------
# The two world views


def scene_xml(episode, size, side, layout):
    walls = grid(episode["walls"], np.uint8, (size, size))
    geoms = []
    for r, c in zip(*np.nonzero(walls)):
        x, y = square_xy(r, c, size)
        geoms.append(f'<geom type="box" pos="{x} {y} 0.5" size="0.5 0.5 0.5" rgba="0.55 0.55 0.55 1"/>')
    for box in episode["boxes"]:
        r, c, h, w = box["rect"]
        top, bottom, left, right = size - r, size - r - h, c, c + w
        for x0, y0, x1, y1 in ((left, top, right, top), (left, bottom, right, bottom), (left, bottom, left, top), (right, bottom, right, top)):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hx, hy = max(abs(x1 - x0) / 2, 0.08), max(abs(y1 - y0) / 2, 0.08)
            geoms.append(f'<geom type="box" pos="{cx} {cy} 0.01" size="{hx} {hy} 0.01" rgba="{rgba(box["rgb"])}"/>')
    border = []
    for x, y, hx, hy in ((size / 2, -0.15, size / 2 + 0.3, 0.15), (size / 2, size + 0.15, size / 2 + 0.3, 0.15), (-0.15, size / 2, 0.15, size / 2), (size + 0.15, size / 2, 0.15, size / 2)):
        border.append(f'<geom type="box" pos="{x} {y} 0.2" size="{hx} {hy} 0.2" rgba="0.2 0.2 0.2 1"/>')
    cheeses = []
    half = side / 2
    for k, cheese in enumerate(episode["cheeses"]):
        cheeses.append(f'<body name="cheese{k}" mocap="true" pos="0 0 {half}"><geom name="cheese{k}" type="box" size="{half} {half} {half}" rgba="{rgba(cheese["rgb"])}"/></body>')
    robots = []
    for k in range(episode["robot_count"]):
        legs = []
        for angle in (90, 210, 330):
            lx, ly = 0.26 * math.cos(math.radians(angle)), 0.26 * math.sin(math.radians(angle))
            legs.append(f'<geom type="capsule" fromto="{lx:.3f} {ly:.3f} 0.02 {lx:.3f} {ly:.3f} 0.24" size="0.035" rgba="0.1 0.1 0.1 1"/>')
        robots.append(
            f'<body name="robot{k}" mocap="true" pos="0 0 0">'
            '<geom type="cylinder" pos="0 0 0.3" size="0.4 0.06" rgba="0.92 0.92 0.92 1"/>'
            '<geom type="cylinder" pos="0 0 0.4" size="0.09 0.05" rgba="0.1 0.1 0.1 1"/>' + "".join(legs) + "</body>"
        )
    return f"""
<mujoco>
  <visual><global offwidth="{layout.width}" offheight="{layout.height}"/><quality shadowsize="{2048 if layout.s < 0.8 else 4096}"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.32 0.32 0.32" rgb2="0 0 0" width="512" height="512"/>
    <texture name="checker" type="2d" builtin="checker" rgb1="0.86 0.86 0.86" rgb2="0.74 0.74 0.74" width="512" height="512"/>
    <material name="floor" texture="checker" texrepeat="16 16"/>
  </asset>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <light pos="{size / 2} {size * 0.2} 40" dir="0 0.3 -1" directional="true" castshadow="true"/>
    <geom type="plane" pos="{size / 2} {size / 2} 0" size="{size * 0.75} {size * 0.75} 0.1" material="floor"/>
    {"".join(border)}
    {"".join(geoms)}
    {"".join(cheeses)}
    {"".join(robots)}
  </worldbody>
</mujoco>"""


class Arena:
    """The episode in MuJoCo's renderer; pose(t) draws robots and cheeses at fractional step t."""

    SHRINK_SECONDS = 0.5  # an eaten cheese shrinks to nothing over this long

    def __init__(self, episode, size, side, frames_per_step, shape, layout):
        self.episode, self.size, self.side = episode, size, side
        self.seconds_per_step = frames_per_step / FPS
        self.model = mujoco.MjModel.from_xml_string(scene_xml(episode, size, side, layout))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, shape[1], shape[0])
        self.camera = mujoco.MjvCamera()
        self.camera.lookat = [size / 2, size / 2 + 0.5, 0]
        self.camera.distance = 42 * max(1.0, (1140 / 870) / (shape[0] / shape[1]))  # back off for narrower views so the arena still fits
        self.camera.azimuth = 90
        self.camera.elevation = -50
        self.robot_ids = [self.model.body(f"robot{k}").mocapid[0] for k in range(episode["robot_count"])]
        self.cheese_ids = [self.model.body(f"cheese{k}").mocapid[0] for k in range(len(episode["cheeses"]))]
        self.cheese_geoms = [self.model.geom(f"cheese{k}").id for k in range(len(episode["cheeses"]))]
        self.eaten_step = {k: step["t"] for step in episode["steps"] for k in step["eaten"]}

    def positions(self, t):
        """(robots (N, 2), cheeses (K, 2)) as continuous (row, col) at fractional step t: the
        recorded physical path when there is one, else straight lines between squares."""
        steps = self.episode["steps"]
        i = min(int(t), len(steps) - 1)
        f = t - i
        n = self.episode["robot_count"]
        if "path" in steps[i]:
            path = np.frombuffer(base64.b64decode(steps[i]["path"]), np.float16).reshape(-1, n + len(self.cheese_ids), 2).astype(float)
            frame = path[min(round(f * len(path)), len(path) - 1)] if i < len(steps) - 1 else path[0]
            return frame[:n], frame[n:]
        j = min(i + 1, len(steps) - 1)
        a, b = robots_of(steps[i])[:, :2], robots_of(steps[j])[:, :2]
        ca = np.array([c[:2] for c in steps[i]["cheeses"]], float).reshape(-1, 2)
        cb = np.array([c[:2] for c in steps[j]["cheeses"]], float).reshape(-1, 2)
        return a * (1 - f) + b * f, ca * (1 - f) + cb * f

    def pose(self, t):
        robots, cheeses = self.positions(t)
        for k, mocap in enumerate(self.robot_ids):
            x, y = square_xy(robots[k, 0], robots[k, 1], self.size)
            self.data.mocap_pos[mocap] = [x, y, 0.0]
        half = self.side / 2
        for k, mocap in enumerate(self.cheese_ids):
            row, col = cheeses[k]
            scale = 1.0
            if k in self.eaten_step:
                elapsed = (t - (self.eaten_step[k] - 1)) * self.seconds_per_step
                scale = float(np.clip(1 - elapsed / self.SHRINK_SECONDS, 0.0, 1.0))
            x, y = square_xy(row + half - 0.5, col + half - 0.5, self.size)
            self.data.mocap_pos[mocap] = [x, y, half * scale - (0 if scale > 0 else 5)]
            self.model.geom_size[self.cheese_geoms[k]] = [max(half * scale, 1e-3)] * 3
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()


class CellView:
    """The grid world drawn flat, one cell per square and no interpolation (each step every robot
    jumps one square or stays), like a cellular automaton."""

    def __init__(self, episode, size, shape):
        self.episode, self.size = episode, size
        self.cell = min(shape) // size
        self.width, self.height = shape
        self.walls = grid(episode["walls"], np.uint8, (size, size))
        self.cheese_rgb = [(np.array(c["rgb"]) * 255).astype(np.uint8) for c in episode["cheeses"]]

    def pose(self, t):
        step = self.episode["steps"][min(int(t), len(self.episode["steps"]) - 1)]
        size, cell = self.size, self.cell
        colours = np.full((size, size, 3), 240, np.uint8)
        colours[self.walls > 0] = 110
        for (r, c, alive), rgb in zip(step["cheeses"], self.cheese_rgb):
            if alive:
                colours[r : r + 4, c : c + 4] = rgb
        robots = robots_of(step)
        colours[robots[:, 0], robots[:, 1]] = 15
        image = Image.fromarray(colours).resize((size * cell, size * cell), Image.NEAREST)
        draw = ImageDraw.Draw(image)
        for k in range(size + 1):
            draw.line((k * cell, 0, k * cell, size * cell), fill=(210, 210, 210), width=1)
            draw.line((0, k * cell, size * cell, k * cell), fill=(210, 210, 210), width=1)
        rim = max(2, cell // 4)
        for box in self.episode["boxes"]:
            r, c, h, w = box["rect"]
            edge = (c * cell, r * cell, (c + w) * cell, (r + h) * cell)
            draw.rectangle((edge[0] - rim // 2, edge[1] - rim // 2, edge[2] + rim // 2, edge[3] + rim // 2), outline=(40, 40, 40), width=rim * 2)  # a white box shows on the light floor
            draw.rectangle(edge, outline=tuple(int(255 * v) for v in box["rgb"]), width=rim)
        frame = Image.new("RGB", (self.width, self.height), "black")
        frame.paste(image, ((self.width - size * cell) // 2, (self.height - size * cell) // 2))
        return np.asarray(frame)


# ---------------------------------------------------------------------------------------------
# Frames


class Video:
    """H.264 through ffmpeg; add(image) appends a frame."""

    def __init__(self, path, layout):
        self.process = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{layout.width}x{layout.height}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def add(self, image, count=1):
        data = image.convert("RGB").tobytes()
        for _ in range(count):
            self.process.stdin.write(data)
        self.frames += count

    def close(self):
        self.process.stdin.close()
        self.process.wait()


def card(lines, layout):
    """A plain black card; lines are (text, size, weight, gap above) at the 1080 reference."""
    image = Image.new("RGB", (layout.width, layout.height), "black")
    draw = ImageDraw.Draw(image)
    faces = [font(layout.px(size), weight) for _, size, weight, _ in lines]
    heights = [face.getbbox("Ag")[3] + layout.px(gap) for (_, _, _, gap), face in zip(lines, faces)]
    y = (layout.height - sum(heights)) / 2
    for (text, _, _, gap), face in zip(lines, faces):
        y += layout.px(gap)
        draw.text(((layout.width - draw.textlength(text, font=face)) / 2, y), text, fill="white", font=face)
        y += face.getbbox("Ag")[3]
    return image


def command_at(episode, i):
    return episode["steps"][i].get("command", 0)


def switches(episode):
    """[(step, command index)]: the step each command takes over at, the first at 0."""
    out = [(0, command_at(episode, 0))]
    for i in range(1, len(episode["steps"])):
        if command_at(episode, i) != command_at(episode, i - 1):
            out.append((i, command_at(episode, i)))
    return out


def typed(episode, t, seconds_per_step):
    """(text, typing): the instruction overlay at fractional step t. A new command is typed in
    letter by letter over TYPE_SECONDS, ending at the step it takes over."""
    commands = episode["commands"]
    lead = TYPE_SECONDS / seconds_per_step  # in steps
    text = commands[command_at(episode, 0)]["text"]
    for at, j in switches(episode)[1:]:
        if t >= at:
            text = commands[j]["text"]
        elif t >= at - lead:
            new = commands[j]["text"]
            return new[: int(len(new) * (t - (at - lead)) / lead)], True
    return text, False


def fitted(draw, text, size, weight, width):
    """The largest font up to size at which text fits in width."""
    while size > 12 and draw.textlength(text, font=font(size, weight)) > width:
        size -= 2
    return font(size, weight)


def compose(worlds, label, t, steering, seconds_per_step, layout):
    """One frame: the instruction on top, the two world views below. worlds: [(episode, view, title)];
    label: top-right text ("Task 2/6"), or ''."""
    L = layout
    image = Image.new("RGB", (L.width, L.height), "black")
    draw = ImageDraw.Draw(image)
    lead = worlds[0][0]
    label_width = 0
    if label:
        face = font(L.px(36))
        label_width = draw.textlength(label, font=face)
        draw.text((L.width - L.px(40) - label_width, L.px(32)), label, fill="white", font=face)
    if steering:
        draw.text((L.px(40), L.px(14)), "Instruction (changed live):", fill=GREY, font=font(L.px(28)))
        text, typing = typed(lead, min(t, len(lead["steps"]) - 1), seconds_per_step)
        shown = f"“{text}" + ("" if typing else "”")
        big = fitted(draw, shown + "”", L.px(56), b"SemiBold", L.width - L.px(140) - label_width)
        draw.text((L.px(40), L.px(50)), shown, fill="white", font=big)
        if typing:  # a text cursor after the letters typed so far
            x = L.px(40) + draw.textlength(shown, font=big) + L.px(6)
            draw.rectangle((x, L.px(64), x + L.px(5), L.px(122)), fill="white")
    else:
        header = f"Instruction: “{lead['commands'][0]['text']}”"
        draw.text((L.px(40), L.px(26)), header, fill="white", font=fitted(draw, header, L.px(44), b"SemiBold", L.width - L.px(140) - label_width))
    top = L.top_steer if steering else L.top
    title_face = font(L.px(30), b"SemiBold")
    for left, (episode, view, title) in zip(L.columns, worlds):
        draw.text((left, top - L.px(44)), title, fill="white", font=title_face)
        if steering:  # each world switches when its own swarm has acted on the command
            i = min(int(t), len(episode["steps"]) - 1)
            status = f"“{episode['commands'][command_at(episode, i)]['text']}”"
            since = max(at for at, _ in switches(episode) if at <= i)
            fresh = since > 0 and (t - since) * seconds_per_step < HIGHLIGHT_SECONDS
            face = fitted(draw, status, L.px(26), b"SemiBold" if fresh else b"Regular", L.view[0] - draw.textlength(title, font=title_face) - L.px(24))
            draw.text((left + L.view[0] - draw.textlength(status, font=face), top - L.px(40)), status, fill="white" if fresh else GREY, font=face)
        image.paste(Image.fromarray(view), (left, top))
        draw.rectangle((left - 1, top - 1, left + L.view[0], top + view.shape[0]), outline="white")
    return image


def render_task(video, episode, partner, label, frames_per_step, layout, size, side, right_title, on_step=None):
    """One task: both worlds advance one policy step per frames_per_step frames; the one that
    finishes first holds its last state. Returns the middle frame (a still). on_step(i, steps), if
    given, is called before each step is drawn."""
    steering = len(episode["commands"]) > 1 and episode.get("steering", False)
    top = layout.top_steer if steering else layout.top
    shape = (layout.view[0], layout.view[1] - (top - layout.top))
    worlds = [(episode, Arena(episode, size, side, frames_per_step, shape, layout), "Physics simulation (MuJoCo)"), (partner, CellView(partner, size, shape), right_title)]
    steps = max(len(e["steps"]) for e, _, _ in worlds) - 1
    still = None
    for i in range(steps + 1):
        if on_step is not None:
            on_step(i, steps)
        sub = frames_per_step if i < steps else int(HOLD_SECONDS * FPS)
        for k in range(sub):
            t = i + (k / frames_per_step if i < steps else 0.0)
            views = [(e, view.pose(min(t, len(e["steps"]) - 1)), title) for e, view, title in worlds]
            frame = compose(views, label, t, steering, frames_per_step / FPS, layout)
            video.add(frame)
            if still is None and i == steps // 2:
                still = frame
    return still


def intro_card(episode, entry, number, total, grid_entry, layout):
    commands = episode["commands"]
    walls = int(grid(episode["walls"], np.uint8, (32, 32)).sum())
    scene = f"{episode['robot_count']} robots, {len(episode['cheeses'])} cheeses, {len(episode['boxes'])} boxes, {'walls' if walls else 'no walls'}"
    lines = [(f"Task {number} of {total}", 36, b"Regular", 0)]
    if entry.get("steering"):
        lines.append(("Real-time steering", 58, b"SemiBold", 24))
        lines.append(("One swarm, no cuts; each instruction is replaced while the swarm is still moving:", 30, b"Regular", 24))
        lines += [(f"{k + 1}. “{c['text']}”", 44, b"SemiBold", 14) for k, c in enumerate(commands)]
        rate = f"The last instruction is completed in {entry['final_successes']}/{entry['rollouts']} rollouts under MuJoCo"
    else:
        lines.append((f"Instruction: “{commands[0]['text']}”", 58, b"SemiBold", 24))
        rate = f"Succeeds in {entry['successes']}/{entry['rollouts']} rollouts under MuJoCo"
    lines.append((scene, 28, b"Regular", 36))
    if grid_entry is not None:
        key = "final_successes" if entry.get("steering") else "successes"
        rate += f" and {grid_entry[key]}/{grid_entry['rollouts']} on the raw grid"
    lines.append((rate + " (fresh scenes).", 28, b"Regular", 10))
    return card(lines, layout)


def shown(data, min_share=0.5):
    """[(episode, entry)]: per instruction its best run, when at least min_share of its rollouts
    succeed (steering: complete the last command) and that run succeeds."""
    out = []
    for entry in data["instructions"]:
        run = entry["runs"][0]
        wins = entry["final_successes"] if entry.get("steering") else entry["successes"]
        if wins >= max(min_share * entry["rollouts"], 1) and run["success"]:
            out.append(({**run, "steering": entry.get("steering", False)}, entry))
        else:
            print("not shown (below min_share, or the best run fails):", entry["instruction"])
    return out


def partner_of(episode, entry, grid_entry):
    """(MuJoCo run, grid run) of one seed that succeeds in both worlds, preferring the best MuJoCo run;
    None when there is none."""
    by_seed = {r["seed"]: r for r in grid_entry["runs"] if r["success"]}
    for run in [episode] + [{**r, "steering": episode["steering"]} for r in entry["runs"] if r["seed"] != episode["seed"]]:
        if run["success"] and run["seed"] in by_seed:
            return run, {**by_seed[run["seed"]], "steering": episode["steering"]}
    return None


def trimmed(episode, after):
    """The episode cut off after steps past the moment its last instruction was met."""
    segments = episode.get("segments")
    if not segments or segments[-1]["met_at"] is None:
        return episode
    end = min(segments[-1]["met_at"] + after, len(episode["steps"]) - 1)
    return {**episode, "steps": episode["steps"][: end + 1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", nargs="+", required=True, help="cellbot.demo.record --physics mujoco outputs; their instructions are shown in order")
    parser.add_argument("--grid", nargs="*", default=[], help="cellbot.demo.record --physics grid outputs for the same instructions: the right panel then shows the same scene on the raw grid")
    parser.add_argument("--out", required=True, help="the .mp4 to write; stills go beside it")
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--trim", type=int, default=4, help="end each run this many steps after its last instruction is met; -1 keeps the holding steps")
    parser.add_argument("--min_share", type=float, default=0.5)
    parser.add_argument("--cards", action=argparse.BooleanOptionalAction, default=True, help="a card with the instruction and its success rate before each task")
    args = parser.parse_args()
    layout = Layout(args.height)
    files = [json.loads(Path(path).read_text()) for path in args.recordings]
    data = {**files[0], "instructions": [entry for f in files for entry in f["instructions"]]}
    grids = {e["instruction"]: e for path in args.grid for e in json.loads(Path(path).read_text())["instructions"]}
    tasks = []
    for episode, entry in shown(data, args.min_share):
        grid_entry = grids.get(entry["instruction"])
        pair = partner_of(episode, entry, grid_entry) if grid_entry else None
        if args.grid and pair is None:
            print("no scene succeeds in both worlds:", entry["instruction"])
        run, partner = pair or (episode, episode)
        if args.trim >= 0:
            run, partner = trimmed(run, args.trim), trimmed(partner, args.trim)
        tasks.append((run, partner, entry, grid_entry if pair else None))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    video = Video(out, layout)
    for number, (run, partner, entry, grid_entry) in enumerate(tasks, 1):
        print("task", number, entry["instruction"], flush=True)
        if args.cards:
            video.add(intro_card(run, entry, number, len(tasks), grid_entry, layout), int(CARD_SECONDS * FPS))
        frames = STEER_FRAMES if run["steering"] else data["frames_per_step"]
        right = "Cellular automaton" if grid_entry else "Cellular automaton (the grid the policy reads)"
        label = f"Task {number}/{len(tasks)}" if len(tasks) > 1 else ""
        still = render_task(video, run, partner, label, frames, layout, data["size"], data["cheese_side"], right)
        still.save(out.with_name(f"{out.stem}-{number:02d}.png"))
    video.close()
    print("wrote", out, round(video.frames / FPS, 1), "s")


if __name__ == "__main__":
    main()
