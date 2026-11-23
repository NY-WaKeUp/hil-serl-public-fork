#!/usr/bin/env python3
"""HIL sim loop: policy explores; human overrides via keyboard when needed."""

from absl import app, flags

from experiments.panda_pick_cube_sim.config import TrainConfig

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "episodes",
    0,
    "Stop after N episodes (0 = run until Ctrl+C).",
)


def main(_):
    config = TrainConfig()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    print("=== HIL 仿真（策略探索 + 人工介入）===")
    print("默认：策略每步随机采样动作，机械臂会自己动。")
    print("介入：按 Space 开启键盘接管，再按 Space 交回给策略。")
    print("  方向键: XY | Shift/Shift_R: Z | 左/右 Ctrl: 夹爪")
    print("  Enter: 本局成功 | ESC: 本局失败")
    print("应弹出 MuJoCo 窗口；若无画面请检查 DISPLAY。")
    print()

    obs, _ = env.reset()
    episode = 0
    step = 0
    ep_return = 0.0
    intvn_steps = 0

    while FLAGS.episodes == 0 or episode < FLAGS.episodes:
        policy_action = env.action_space.sample()
        next_obs, rew, done, truncated, info = env.step(policy_action)
        reward = float(rew)
        step += 1
        ep_return += reward
        if info.get("is_intervention", False):
            intvn_steps += 1

        obs = next_obs
        if step % 10 == 0:
            mode = "人工" if info.get("is_intervention") else "策略"
            print(f"step={step} mode={mode} ep_return={ep_return:.2f}", end="\r")

        if done or truncated:
            episode += 1
            succeeded = bool(
                reward > 0 or info.get("succeed") or info.get("next.success")
            )
            print(
                f"\nEpisode {episode}: return={ep_return:.2f} "
                f"success={succeeded} intervention_steps={intvn_steps}"
            )
            ep_return = 0.0
            intvn_steps = 0
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    app.run(main)
