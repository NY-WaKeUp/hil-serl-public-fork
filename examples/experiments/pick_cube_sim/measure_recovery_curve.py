# Recovery-curve measurement: evaluate all checkpoints in a run directory
# under the fixed perturbed lighting, producing SR(step) for the recovery plot.
# Usage:
#   python measure_recovery_curve.py --run_dir .../recover_mech_template \
#       --n_episodes 20 --out_csv recovery_mech.csv

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))      # examples/
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))  # repo root

import csv
import glob
import re

from absl import app, flags

from experiments.pick_cube_sim.eval_common import eval_preset, load_agent_at, make_env_and_agent
from experiments.pick_cube_sim.lighting_utils import LightingConfig

FLAGS = flags.FLAGS
flags.DEFINE_string("run_dir", None, "Checkpoint dir of a recovery run.")
flags.DEFINE_integer("n_eps", 10, "Episodes per checkpoint.")
flags.DEFINE_string("out_csv", "recovery.csv", "Output CSV.")
flags.mark_flag_as_required("run_dir")


def main(_):
    env, agent, _config = make_env_and_agent()
    model = env.unwrapped.model
    cfg = LightingConfig(diffuse_scale=0.1)  # matches retrain lighting (0.1)

    steps = sorted(
        int(re.search(r"checkpoint_(\d+)", p).group(1))
        for p in glob.glob(os.path.join(FLAGS.run_dir, "checkpoint_*"))
    )
    print(f"Found checkpoints at steps: {steps}")

    rows = []
    for step in steps:
        agent_eval = load_agent_at(agent, FLAGS.run_dir, step)
        sr, (lo, hi), t = eval_preset(agent_eval, env, model, cfg, FLAGS.n_eps, seed=0)
        print(f"step {step:>7d}  SR={sr:5.1%}  CI95=[{lo:.3f},{hi:.3f}]  t={t:.1f}s")
        rows.append({"step": step, "success_rate": sr, "ci_lo": lo, "ci_hi": hi,
                     "avg_time": t})

    with open(FLAGS.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "success_rate", "ci_lo", "ci_hi",
                                               "avg_time"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {FLAGS.out_csv}")


if __name__ == "__main__":
    app.run(main)
