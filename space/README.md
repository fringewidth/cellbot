---
title: cellbot
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
license: apache-2.0
short_description: One language model steers a robot swarm from plain English
models:
  - fringewidth/cellbot1
  - Qwen/Qwen3-0.6B
---

# cellbot

The demo of [cellbot1](https://huggingface.co/fringewidth/cellbot1): one language model steering a
swarm of about 100 robots from plain English, shown in MuJoCo beside the grid it reads.
Code: [github.com/fringewidth/cellbot](https://github.com/fringewidth/cellbot).

This folder is the Space's source. `cellbot1.mp4` is rendered by `cellbot.demo.render` and uploaded
with it.
