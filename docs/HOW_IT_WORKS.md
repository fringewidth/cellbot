# How cellbot1 works

## The world

`cellbot.sim.world` is a batched grid world in PyTorch. A scene has a 32 x 32 grid with at most one
robot per square, terrain (walls), up to 6 cheeses of 4 x 4 squares and up to 3 boxes, which are
rectangles drawn on the floor. Cheeses and boxes have colours: red, green, blue, yellow, orange,
purple, white or cyan in training. Pink is held out: it appears only in tests, to check a colour
word the policy never saw.

Each step every robot picks one of 9 moves: one of the 8 neighbours, or stay. Moves into walls or
occupied squares are resolved by the world. A cheese moves one square when the robots pushing it
in one direction add up to at least 3, and it is eaten when enough of its ring (the squares
around it) is covered by robots.

The policy sees 10 channels: robots, walls, cheese mask, cheese colour (RGB), box mask, box colour
(RGB).

`cellbot.sim.mujoco_world` runs the same scenes as rigid-body physics. Robots are flat cylinders
with velocity-limited actuators. Cheeses are boxes with dry friction between two and three robots'
force, so pushing takes three robots, as on the grid. Each policy step sets every robot's target to
the neighbouring square its move points to and simulates one second. The scene is then snapped back
to the grid by a Hungarian assignment of robots to free squares. Eating uses the grid's rule on the
snapped state.

## The policy

`cellbot.policy.model.LMPolicy` wraps a pretrained causal LM (Qwen3-0.6B):

1. The instruction is tokenised as usual and embedded by the LM's own embedding table.
1. The grid is cut into 64 patches of 4 x 4 squares. Each patch (10 channels x 16 squares) is one
   linear projection into the LM's hidden size, plus learned row and column embeddings.
1. Text and scene tokens go through the LM's transformer together. Text tokens attend causally to
   text only, so the LM's language processing is exactly what it was. Scene tokens attend to every
   token. All scene tokens sit at one rotary position, just after the text; their place on the grid
   comes only from the row and column embeddings.
1. A linear readout maps each scene token's final hidden state to 16 x 9 logits: a categorical over
   the 9 moves for each square of its patch.
1. Every square samples its move independently, every step. Robots use the move of the square they
   stand on; empty squares' moves are ignored.
1. Classifier-free guidance: the same scene with the text masked out gives the null logits, and
   the policy samples from `null + 2.0 * (conditioned - null)`. The null instruction was trained
   by dropping the text in 10% of examples.

The trainable parts are the rank-64 LoRA on every attention and MLP projection and the new
layers (patch embeddings, position tables, readout): about 41M parameters in all, the whole
`cellbot1.safetensors`. The base LM's weights are frozen, including its embeddings.

### Why a plain readout

Before the LM, a custom transformer was trained from scratch (its code is kept in
`cellbot.training.custom_model`). Its action head was compared as a pi0-style flow-matching
expert, a masked discrete-diffusion expert and a plain linear readout from the backbone. The readout
won clearly: on held-out play programs it closed 84% of the gap to replaying the recorded actions,
against 61% for the flow expert, and it recalled 44% of the cheese-eating events against 18%. It
was also about 2.4 times faster per step. Per-square categorical outputs close a cheese's ring.
Smooth velocity fields from a flow model tend not to.

## Data

1. **Play.** Small Python programs written by LLM agents (the brief is
   `training/PLAY_BRIEF.md`) drive the shared controller with up to 4 attractor points per step and mark their phases. Their episodes are cut
   into windows at phase boundaries. Claude Haiku then captioned each window with four
   instructions (short, medium, detailed, casual) from its facts and text renderings of the grid
   (`training/caption.py`). The first stage of training learns from these captions and from goal
   frames.
1. **Experts.** For each task family a scripted expert turns the instruction and scene into
   attractors, for example "the free squares of the red cheese's ring", then "press in". The shared
   controller (`experts/controller.py`) turns the attractors into a move for every square by
   distance fields. `experts/attractor.py` writes stores of these episodes. They are the bulk of
   the last two stages.
1. **General text.** Batches of general text, instruction data and GSM8K problems, on which the
   LM with LoRA is distilled toward the frozen original, so the model keeps its language and
   arithmetic.

## What did not work

1. **RL on robot actions** (`training/rl.py`). GRPO-style group baselines with the task checkers as
   reward. Training success stayed flat at 0.16-0.20 over about 200 updates on Kaggle. One scalar
   reward for about 6,000 robot actions per rollout gives too little signal.
1. **Language-level plans.** An LM writes a sequence of sub-instructions and the policy executes
   them. Even with perfect (oracle) plans, held-out logic tasks reached only 0.10-0.33, because the
   executor eats forbidden cheeses on the way in about 40% of episodes. The executor caps the plan.
1. **Zero-shot planners.** Off-the-shelf small Qwen models as planners were weak.

## Known weak spots

1. Cheeses in corners and pockets of the terrain are hard to push or surround.
1. The policy cannot reliably push a cheese to a target (the herd family), so the live
   suggestions in `cellbot.steer` leave that family out.
