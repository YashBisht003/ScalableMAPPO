import os
import numpy as np
import torch

from WarehouseEnv import WarehouseEnv
from scalablemappo import ScalableMAPPO
from video_recorder import VideoRecorder
from configs import CONFIG


CHECKPOINT_PATH = "./checkpoints/final_model.pt"
VIDEO_PATH = "./videos/mappo_play.mp4"


def main():

    os.makedirs("videos", exist_ok=True)

    print("Loading environment...")

    env = WarehouseEnv(
        height=CONFIG['height'],
        width=CONFIG['width'],
        n_robots=CONFIG['n_robots'],
        max_steps=CONFIG['max_steps'],
        view_size=CONFIG['view_size'],
        render_mode="rgb_array"   # IMPORTANT
    )

    obs, _ = env.reset()

    print("Loading Scalable MAPPO model...")

    global_state = env.get_global_state()

    mappo = ScalableMAPPO(
        obs_channels=obs["robot_0"].shape[0],
        view_size=obs["robot_0"].shape[1],
        n_actions=8,                         # CHARGE action
        n_agents=CONFIG['n_robots'],
        global_state_dim=len(global_state)
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    mappo.load(CHECKPOINT_PATH)

    print("Video recording ON...")

    recorder = VideoRecorder(VIDEO_PATH, fps=10)

    done = False
    step = 0
    total_reward = 0

    while not done:

        global_state = env.get_global_state()

        actions = {}

        for agent in env.agents:
            agent_id = int(agent.split("_")[1])

            action, _, _ = mappo.select_action(
                obs[agent],
                global_state,
                agent_id,
                deterministic=True
            )
            actions[agent] = action

        obs, rewards, terms, truncs, _ = env.step(actions)

        # ---- Capture frame ----
        frame = env.render()
        if frame is not None:
            recorder.add_frame(frame)

        total_reward += np.mean(list(rewards.values()))
        done = any(terms.values()) or any(truncs.values())
        step += 1

    recorder.close()
    env.close()

    print("=" * 60)
    print("Playback finished!")
    print(f"Steps: {step}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Video saved to: {VIDEO_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
