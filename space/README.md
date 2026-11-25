---
title: cellbot
colorFrom: gray
colorTo: gray
sdk: static
app_file: index.html
license: apache-2.0
short_description: A language model steers a robot swarm, live in your browser
models:
  - fringewidth/cellbot1
  - Qwen/Qwen3-0.6B
---

# cellbot

The live demo of [cellbot1](https://huggingface.co/fringewidth/cellbot1). You type instructions; a
post-trained Qwen3-0.6B moves every robot of your own swarm in MuJoCo; your browser draws it with
three.js. The model and the physics run on Modal (`deploy/modal_app.py` in
[the code](https://github.com/fringewidth/cellbot)), and this Space serves the page (`site/`).
