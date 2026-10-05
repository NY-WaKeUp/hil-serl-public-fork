#!/usr/bin/env python3
"""Offline-evaluated, systemd-backed diffuse lighting curriculum controller."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent))

from experiments.pick_cube_sim.eval_common import (
    eval_preset,
    load_agent_at,
    make_env_and_agent,
)
from experiments.pick_cube_sim.lighting_utils import LightingConfig

BASE = _HERE
LAUNCHER = BASE / "train_course.sh"
CHECKPOINT_PERIOD = 5000


def stage_label(diffuse: float) -> str:
    return f"{diffuse:g}"


def stage_run_dir(arm: str, seed: int, diffuse: float) -> Path:
    return BASE / f"course_{arm}_diffuse{stage_label(diffuse)}_seed{seed}"


def manifest_path(arm: str, seed: int) -> Path:
    return BASE / f"course_{arm}_seed{seed}_manifest.json"


def atomic_write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def checkpoint_steps(run_dir: Path) -> list[int]:
    return sorted(
        int(path.name.removeprefix("checkpoint_"))
        for path in run_dir.glob("checkpoint_*")
        if path.is_dir() and path.name.removeprefix("checkpoint_").isdigit()
    )


def unit_name(arm: str, seed: int) -> str:
    return f"hilserl-course-{arm}-seed{seed}-train.service"


def unit_active(arm: str, seed: int) -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit_name(arm, seed)]
    ).returncode == 0


def launcher(command: str, args: argparse.Namespace, diffuse: float, source: dict | None = None) -> None:
    cmd = ["bash", str(LAUNCHER), command, args.arm, str(args.seed), stage_label(diffuse), str(args.tau)]
    if source is not None:
        cmd.extend([source["run_dir"], str(source["step"])])
    subprocess.run(cmd, check=True)


def evaluate(agent, env, run_dir: Path, step: int, diffuse: float, seed: int, n_eps: int) -> dict:
    policy = load_agent_at(agent, str(run_dir), step)
    sr, (lb, ub), mean_time = eval_preset(
        policy,
        env,
        env.unwrapped.model,
        LightingConfig(diffuse_scale=diffuse),
        n_eps,
        seed,
    )
    return {"step": step, "eval_seed": seed, "sr": sr, "lb": lb, "ub": ub, "mean_time": mean_time}


def new_manifest(args: argparse.Namespace, stages: list[float]) -> dict:
    return {
        "version": 1,
        "arm": args.arm,
        "seed": args.seed,
        "tau": args.tau,
        "target_lower_bound": args.target_lb,
        "n_episodes": args.n_eps,
        "max_stage_steps": args.max_stage_steps,
        "stages": stages,
        "current_index": 0,
        "source": {"run_dir": str(BASE / "first_run"), "step": 30000},
        "phases": [],
        "status": "running",
    }


def load_manifest(args: argparse.Namespace, stages: list[float]) -> dict:
    path = manifest_path(args.arm, args.seed)
    if path.exists():
        return json.loads(path.read_text())
    manifest = new_manifest(args, stages)
    atomic_write_json(path, manifest)
    return manifest


def phase_for(manifest: dict, diffuse: float) -> dict | None:
    return next((item for item in manifest["phases"] if item["diffuse"] == diffuse), None)


def stage_passed(phase: dict) -> int | None:
    passing = [item for item in phase["evaluations"] if item["lb"] >= phase["target_lb"]]
    for first in passing:
        for second in passing:
            if first["eval_seed"] != second["eval_seed"] and second["step"] - first["step"] >= CHECKPOINT_PERIOD:
                return second["step"]
    return None


def evaluate_new_checkpoints(args: argparse.Namespace, phase: dict, agent, env) -> None:
    seen = {(item["step"], item["eval_seed"]) for item in phase["evaluations"]}
    for step in checkpoint_steps(Path(phase["run_dir"])):
        # Alternate seeds per checkpoint. Passing needs distinct seeds and steps.
        eval_seed = 0 if (step // CHECKPOINT_PERIOD) % 2 == 0 else 1
        if (step, eval_seed) in seen:
            continue
        record = evaluate(agent, env, Path(phase["run_dir"]), step, phase["diffuse"], eval_seed, args.n_eps)
        phase["evaluations"].append(record)
        print(
            f"stage {phase['diffuse']:g} checkpoint_{step} seed={eval_seed}: "
            f"SR={record['sr']:.1%}, WilsonLB={record['lb']:.3f}"
        )


def entry_evaluation(args: argparse.Namespace, source: dict, diffuse: float, agent, env) -> list[dict]:
    source_dir = Path(source["run_dir"])
    return [
        evaluate(agent, env, source_dir, source["step"], diffuse, eval_seed, args.n_eps)
        for eval_seed in (0, 1)
    ]


def tick(args: argparse.Namespace, manifest: dict, agent, env) -> bool:
    idx = manifest["current_index"]
    if idx >= len(manifest["stages"]):
        manifest["status"] = "completed"
        return False

    diffuse = manifest["stages"][idx]
    phase = phase_for(manifest, diffuse)
    if phase is None:
        entry = entry_evaluation(args, manifest["source"], diffuse, agent, env)
        skipped = all(item["lb"] >= args.target_lb for item in entry)
        phase = {
            "diffuse": diffuse,
            "source": manifest["source"].copy(),
            "run_dir": str(stage_run_dir(args.arm, args.seed, diffuse)),
            "target_lb": args.target_lb,
            "entry_evaluations": entry,
            "evaluations": [],
            "status": "skipped" if skipped else "pending",
        }
        manifest["phases"].append(phase)
        print(f"entry diffuse={diffuse:g}: " + ", ".join(f"seed{x['eval_seed']} LB={x['lb']:.3f}" for x in entry))
        if skipped:
            manifest["current_index"] += 1
            print(f"skip diffuse={diffuse:g}: incoming policy already passes")
            return True

    if phase["status"] == "pending":
        if not args.apply:
            print(f"DRY RUN: would start diffuse={diffuse:g} from {phase['source']}")
            return False
        launcher("start", args, diffuse, phase["source"])
        phase["status"] = "running"
        print(f"started diffuse={diffuse:g}")
        return True

    if phase["status"] != "running":
        return False

    if not unit_active(args.arm, args.seed):
        phase["status"] = "interrupted"
        print(f"stage diffuse={diffuse:g} interrupted: training unit is inactive")
        return False

    evaluate_new_checkpoints(args, phase, agent, env)
    winner = stage_passed(phase)
    if winner is not None:
        phase["status"] = "passed"
        phase["winner"] = winner
        manifest["source"] = {"run_dir": phase["run_dir"], "step": winner}
        manifest["current_index"] += 1
        if args.apply:
            launcher("stop", args, diffuse)
        print(f"passed diffuse={diffuse:g} at checkpoint_{winner}")
        return True

    latest = max(checkpoint_steps(Path(phase["run_dir"])), default=phase["source"]["step"])
    if latest - phase["source"]["step"] >= args.max_stage_steps:
        phase["status"] = "failed"
        phase["failure_step"] = latest
        manifest["status"] = "failed"
        if args.apply:
            launcher("stop", args, diffuse)
        print(f"failed diffuse={diffuse:g}: stage budget exhausted at checkpoint_{latest}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("mech", "vanilla"), default="mech")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=3000)
    parser.add_argument("--stages", default="0.8,0.6,0.4,0.2,0.1")
    parser.add_argument("--target-lb", type=float, default=0.70)
    parser.add_argument("--n-eps", type=int, default=20)
    parser.add_argument("--max-stage-steps", type=int, default=50000)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("DISPLAY", ":10")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    stages = [float(value) for value in args.stages.split(",")]
    manifest = load_manifest(args, stages)
    env, agent, _ = make_env_and_agent()
    path = manifest_path(args.arm, args.seed)

    while True:
        changed = tick(args, manifest, agent, env)
        atomic_write_json(path, manifest)
        print(f"manifest={path} status={manifest['status']} changed={changed}")
        if not args.watch or manifest["status"] in ("completed", "failed"):
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
