"""The action space: every square picks one of 9 moves per step (8 directions or stay).

A move field is (..., 2, H, W): the unit vector each square moves along. Its compact form is a code per
square (..., H, W): 0-7 for the unit vector toward world.DIRS[k], 8 for stay.
"""

import math

import torch

from cellbot.sim.world import DIRS

N_ACTIONS = 9
CHUNK = 8  # action steps a decision covers (the policy samples only the first; kept for the data format)
EXECUTE = 4  # steps executed before deciding again (the play-data trainers)
CODE_VECTORS = torch.tensor([[dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)] for dy, dx in DIRS] + [[0.0, 0.0]])


def encode_field(field):
    """(..., 2, H, W) move field -> (..., H, W) int8 code."""
    vx, vy = field[..., 0, :, :], field[..., 1, :, :]
    k = torch.remainder(torch.round(torch.atan2(vy, vx) / (math.pi / 4)), 8)
    zero = torch.sqrt(vx * vx + vy * vy) < 0.5
    return torch.where(zero, 8, k).to(torch.int8)


def decode_field(code, device=None):
    """(..., H, W) int8 code -> (..., 2, H, W) float move field."""
    table = CODE_VECTORS.to(code.device if device is None else device)
    return table[code.long().to(table.device)].movedim(-1, -3)


def patchify(x, patch):
    """(B, C, G, G) -> (B, (G / patch)^2, C * patch * patch), patches in row-major order."""
    b, c, g, _ = x.shape
    n = g // patch
    return x.view(b, c, n, patch, n, patch).permute(0, 2, 4, 1, 3, 5).reshape(b, n * n, c * patch * patch)
