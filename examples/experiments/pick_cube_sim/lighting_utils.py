# Lighting perturbation utilities for PandaPickCube sim.
# Applies deterministic lighting configs directly to mjModel fields shared by
# both the offscreen renderer and the human viewer, so no env wrapper is needed.

import dataclasses
from typing import Tuple

import mujoco
import numpy as np


@dataclasses.dataclass(frozen=True)
class LightingConfig:
    # Multiplicative factor on both directional lights' diffuse color.
    diffuse_scale: float = 1.0
    # Headlight ambient (scene <visual><headlight ambient>), baseline 0.5.
    ambient: float = 0.5
    # Additive offset applied to light[0] (overhead light, baseline pos [0,0,3]).
    light0_pos_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Per-channel scale on light diffuse to emulate color temperature shift.
    color_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def label(self) -> str:
        return (
            f"diffuse{self.diffuse_scale:g}_amb{self.ambient:g}"
            f"_pos{''.join(str(v) for v in self.light0_pos_offset)}"
            f"_col{''.join(str(v) for v in self.color_scale)}"
        )


# Baseline values read from arena.xml (2 directional lights + headlight).
_BASE_LIGHT0_DIFFUSE = np.array([1.0, 1.0, 1.0])
_BASE_LIGHT1_DIFFUSE = np.array([0.8, 0.8, 0.8])
_BASE_LIGHT0_POS = np.array([0.0, 0.0, 3.0])
_BASE_HEADLIGHT_AMBIENT = 0.5


def apply_lighting(model: mujoco.MjModel, cfg: LightingConfig) -> None:
    """Mutate model lighting in place. Safe to call before any render."""
    model.light_diffuse[0] = _BASE_LIGHT0_DIFFUSE * np.asarray(cfg.color_scale) * cfg.diffuse_scale
    model.light_diffuse[1] = _BASE_LIGHT1_DIFFUSE * np.asarray(cfg.color_scale) * cfg.diffuse_scale
    model.light_pos[0] = _BASE_LIGHT0_POS + np.asarray(cfg.light0_pos_offset)
    # Baseline headlight ambient is 0.5 (arena.xml <headlight ambient=".5 .5 .5">);
    # MjVisual in MuJoCo 2.3.7 stores it under `headlight`, not `light`.
    model.vis.headlight.ambient = _BASE_HEADLIGHT_AMBIENT * cfg.ambient / 0.5


def snapshot_lighting(model: mujoco.MjModel) -> dict:
    return {
        "light_diffuse": model.light_diffuse.copy(),
        "light_pos": model.light_pos.copy(),
        "headlight_ambient": float(model.vis.light.ambient),
    }


def restore_lighting(model: mujoco.MjModel, snap: dict) -> None:
    model.light_diffuse[:] = snap["light_diffuse"]
    model.light_pos[:] = snap["light_pos"]
    model.vis.light.ambient = snap["headlight_ambient"]


# Preset sweep: axis A (diffuse intensity) first; C (position) and D (color)
# enabled only if intensity alone cannot break the policy.
PRESETS = [
    LightingConfig(diffuse_scale=1.0),   # baseline sanity check
    LightingConfig(diffuse_scale=0.8),
    LightingConfig(diffuse_scale=0.6),
    LightingConfig(diffuse_scale=0.4),
    LightingConfig(diffuse_scale=0.2),
    LightingConfig(diffuse_scale=0.1),
    LightingConfig(diffuse_scale=0.05),
    # Axis C: move overhead light far sideways -> strong shadow shift.
    LightingConfig(light0_pos_offset=(1.5, 0.0, -1.0)),
    LightingConfig(light0_pos_offset=(-1.5, 0.0, -1.0)),
    # Axis D: warm/cool color shift.
    LightingConfig(color_scale=(1.3, 0.9, 0.6)),
    LightingConfig(color_scale=(0.6, 0.9, 1.3)),
    # Axis A+C combined: dim + shifted light.
    LightingConfig(diffuse_scale=0.2, light0_pos_offset=(1.0, 0.0, -0.5)),
]
