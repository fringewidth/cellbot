# How cellbot1 was trained

A chain of runs, each starting from the previous one's checkpoint: first on play data, then on expert
demonstrations. All ran on Kaggle T4s, about 40 GPU hours in total.
The base model is Qwen3-0.6B with a rank-64 LoRA (alpha 128) on every attention and MLP projection.
Its embeddings stay frozen.

## Play data (`training/lm_train.py`)

The policy learns from play episodes: small programs written by LLM agents drive the swarm
through the shared controller (`training/play.py`, `training/play_cli.py`). Their episodes are cut
into windows at phase boundaries, and each window is captioned by Claude Haiku with four
instructions of different lengths (`training/caption.py`, `training/CAPTION_BRIEF.md`). The policy
is trained to predict the recorded moves from the caption, or from the window's end frame as a goal
image. At the same time the LM with LoRA is distilled toward the frozen original on general text,
instruction data and GSM8K problems, so it keeps its language. In 10% of examples the conditioning
is dropped, which trains the null instruction used for classifier-free guidance.

## Expert demonstrations (`training/expert_sft.py`)

Behaviour cloning on the scripted experts' episodes (`experts/attractor.py`) over the whole task
suite: logic, place, formation, runner, herd and parallel families. Each batch is 75% expert
demonstrations and 25% play captions, plus the distillation batches. The last run, from the
config shipped with the weights:

| setting | value |
| --- | --- |
| batch | 16 demonstration windows (+ captions and 2 distillation sequences of 256 tokens) |
| learning rate | 1e-4 on the LoRA, 2e-4 on the new layers, 100 warmup steps |
| weight decay | 0.05 |
| distillation weight | 1.0 |
| null-instruction rate | 0.1 |
| length | 480 minutes on one T4, 29,177 steps |

Expert stores are generated, not downloaded:

```sh
uv run python -m cellbot.experts.attractor --out data/experts --episodes 20000
uv run python -m cellbot.training.expert_sft overfit --tasks 4 --steps 300                 # the learning gate: can it fit 4 tasks?
uv run python -m cellbot.training.expert_sft train --store data/experts --init_run DIR --name my-run
```

`--init_run` takes a directory with `config.json` and the checkpoint, such as a download of the
Hugging Face repo with `--checkpoint cellbot1.safetensors`. Training from scratch also needs the play
cache and captions (`--play_cache`, `--captions`). The play episodes and captions are not
published yet.

## Final evaluation

From the last run's training log:

| measure | start of the run | end |
| --- | --- | --- |
| held-out demonstration loss | 0.408 | 0.307 |
| "eat the X cheese", scenes with 2+ colours: X eaten / another eaten | 0.87 / 0.12 | 0.91 / 0.09 |
| "go into the X box", scenes with 2+ boxes: into X / into another | 0.19 / - | 0.22 / 0.16 |
| "eat the X cheese" with no X in the scene: something else eaten | - | 0.25 |
| perplexity on held-out general text | 28.0 | 28.2 |

The grounding scenes come from play data, not from the task suite. Going into a named box is the
weakest grounding result.

Fresh tasks from the suite's test split scored 0.82 (place), 0.57 (formation) and 0.46 (logic),
where 1.0 means every task fully met. The demo video's instructions were chosen from the families that
succeed in at least half their rollouts.
