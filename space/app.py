"""The cellbot Space: the demo video, and cellbot1 run on demand on the Space's CPU."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")  # software rendering; must be set before mujoco loads
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import tempfile  # noqa: E402

import gradio as gr  # noqa: E402
import torch  # noqa: E402

from cellbot.demo.live import VIDEO_TASKS, Runner, save_video  # noqa: E402

FREE_TEXT = "Your own words"
MAX_COMMANDS = 3
HEIGHT = 540

torch.set_num_threads(os.cpu_count() or 2)
runner = Runner(device="cpu")

INTRO = """
# cellbot1

One language model steers a swarm of about 100 robots from plain English. Each step it reads the
instruction and the whole 32 x 32 grid, and gives every cell a move: Qwen3-0.6B with a LoRA, trained
by behaviour cloning on scripted experts. Left: the swarm in MuJoCo physics. Right: the grid the
policy reads, a cellular automaton.

[Code](https://github.com/fringewidth/cellbot) · [Weights](https://huggingface.co/fringewidth/cellbot1)
"""

HOW = """
Pick a task from the video, or type your own instruction. Several lines are given one after another,
like changing the command mid-run. Tasks from the video end once their checker says they are done;
your own words run for a fixed number of steps each and are not scored. A run takes one to three
minutes on this Space's two CPU cores.
"""


def summary(episode):
    lines = []
    for s in episode["segments"]:
        steps = s["end"] - s["start"]
        if s["success"] is None:
            outcome = f"ran {steps} steps"
        elif s["success"]:
            outcome = f"done at step {s['met_at']}"
        else:
            outcome = f"not met (score {s['final_score']}) after {steps} steps"
        eaten = f"; ate {', '.join(s['eaten'])}" if s["eaten"] else ""
        lines.append(f"- “{s['text']}”: {outcome}{eaten}")
    return f"{episode['robot_count']} robots.\n\n" + "\n".join(lines)


def run(choice, text, seed, steps, progress=gr.Progress()):
    seed = None if seed is None else int(seed)
    if choice == FREE_TEXT:
        commands = [line.strip() for line in (text or "").splitlines() if line.strip()]
        if not commands:
            raise gr.Error("Type an instruction first.")
        if len(commands) > MAX_COMMANDS:
            raise gr.Error(f"At most {MAX_COMMANDS} instructions per run.")
        total = len(commands) * int(steps)
        tick = lambda t: progress(min(t / total, 1.0), desc=f"Policy step {t}")  # noqa: E731
        episode = runner.free(commands, seed or 0, int(steps), tick)
    else:
        tick = lambda t: progress(min(t / 140, 1.0), desc=f"Policy step {t}")  # noqa: E731
        episode = runner.task(choice, seed, tick)
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    save_video(episode, path, HEIGHT, on_step=lambda i, n: progress(i / max(n, 1), desc=f"Rendering step {i} of {n}"))
    return path, summary(episode)


with gr.Blocks(title="cellbot1", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(INTRO)
    gr.Video("cellbot1.mp4", label="The demo (selected successful runs; success rates are in the video)", autoplay=True, loop=True, interactive=False)
    gr.Markdown("## Run it")
    gr.Markdown(HOW)
    with gr.Row():
        with gr.Column():
            choice = gr.Radio([*VIDEO_TASKS, FREE_TEXT], value=FREE_TEXT, label="Task")
            text = gr.Textbox("eat the red cheese", lines=3, label="Your own words (one instruction per line)")
            with gr.Row():
                seed = gr.Number(None, precision=0, label="Scene seed (blank: the video's scene, or 0)")
                steps = gr.Slider(20, 120, 60, step=10, label="Steps per instruction (your own words)")
            button = gr.Button("Run", variant="primary")
        with gr.Column():
            video = gr.Video(label="Your run", autoplay=True, interactive=False)
            result = gr.Markdown()
    button.click(run, [choice, text, seed, steps], [video, result])

demo.queue(default_concurrency_limit=1, max_size=16).launch()
