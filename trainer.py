import numpy as np
import torch
from typing import Dict
import time
from collections import deque

from WarehouseEnv import WarehouseEnv
from scalablemappo import ScalableMAPPO


class MAPPOTrainer:
    """
    Training loop for Scalable MAPPO in warehouse environment.
    """

    def __init__(
            self,
            env: WarehouseEnv,
            mappo: ScalableMAPPO,
            n_steps: int = 4096,
            n_epochs: int = 4,
            batch_size: int = 128,
            save_interval: int = 50,
            log_interval: int = 10,
            save_dir: str = "./checkpoints"
    ):
        self.env = env
        self.mappo = mappo
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.save_dir = save_dir

        # Metrics
        self.episode_rewards = {agent: [] for agent in env.possible_agents}
        self.episode_lengths = []
        self.episode_deliveries = []
        self.episode_collisions = []

        self.reward_window = deque(maxlen=100)
        self.delivery_window = deque(maxlen=100)

    # --------------------------------------------------------
    # Rollout Collection
    # --------------------------------------------------------

    def collect_rollouts(self):

        step_count = 0
        episode_count = 0

        obs, _ = self.env.reset()
        episode_rewards = {agent: 0 for agent in self.env.agents}
        episode_steps = 0

        while step_count < self.n_steps:

            global_state = self.env.get_global_state()

            actions = {}
            agent_data = {}

            # -------- ACTION SELECTION -------- #
            for agent in self.env.agents:
                agent_id = int(agent.split("_")[1])

                action, log_prob, value = self.mappo.select_action(
                    obs[agent],
                    global_state,
                    agent_id
                )

                actions[agent] = action
                agent_data[agent] = {
                    "obs": obs[agent],
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                    "agent_id": agent_id
                }

            # -------- ENV STEP -------- #
            next_obs, rewards, terms, truncs, infos = self.env.step(actions)

            # Render if needed
            if self.env.render_mode == "human":
                self.env.render()

            # -------- STORE IN BUFFER -------- #
            for agent in self.env.agents:
                done = terms[agent] or truncs[agent]

                self.mappo.buffer.add(
                    obs=agent_data[agent]["obs"],
                    action=agent_data[agent]["action"],
                    reward=rewards[agent],
                    done=done,
                    value=agent_data[agent]["value"],
                    log_prob=agent_data[agent]["log_prob"],
                    global_state=global_state,
                    agent_id=agent_data[agent]["agent_id"]
                )

                episode_rewards[agent] += rewards[agent]

            step_count += 1
            episode_steps += 1

            # -------- EPISODE END -------- #
            if any(terms.values()) or any(truncs.values()):

                avg_reward = np.mean(
                    [episode_rewards[a] for a in self.env.agents]
                )

                self.reward_window.append(avg_reward)
                self.episode_lengths.append(episode_steps)

                for a in self.env.agents:
                    self.episode_rewards[a].append(episode_rewards[a])

                metrics = self.env.world.get_metrics()

                self.episode_deliveries.append(metrics["total_deliveries"])
                self.delivery_window.append(metrics["total_deliveries"])
                self.episode_collisions.append(metrics["collision_count"])

                # Reset
                obs, _ = self.env.reset()
                episode_rewards = {agent: 0 for agent in self.env.agents}
                episode_steps = 0
                episode_count += 1
            else:
                obs = next_obs

        return episode_count

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    def train(self, total_timesteps: int):

        print("=" * 70)
        print("Starting Scalable MAPPO Training")
        print("=" * 70)

        timesteps = 0
        update_count = 0
        start_time = time.time()

        while timesteps < total_timesteps:

            print(f"\n[Update {update_count + 1}] Collecting rollouts...")
            episode_count = self.collect_rollouts()
            timesteps += self.n_steps

            print(f"[Update {update_count + 1}] Updating networks...")
            train_metrics = self.mappo.update(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size
            )

            update_count += 1

            # -------- LOGGING -------- #
            if update_count % self.log_interval == 0:

                elapsed = time.time() - start_time
                fps = timesteps / elapsed

                print("\n" + "=" * 70)
                print(f"Update {update_count}")
                print(f"Timesteps: {timesteps:,}")
                print(f"FPS: {fps:.1f}")

                if self.reward_window:
                    print(f"Mean Reward(100): {np.mean(self.reward_window):.2f}")

                if self.delivery_window:
                    print(f"Mean Deliveries(100): {np.mean(self.delivery_window):.2f}")

                print("Losses:")
                print(f"Policy: {train_metrics['policy_loss']:.4f}")
                print(f"Value: {train_metrics['value_loss']:.4f}")
                print(f"Entropy: {train_metrics['entropy']:.4f}")

                # Agent entropy stats
                for k, v in train_metrics.items():
                    if "agent_" in k:
                        print(f"{k}: {v:.4f}")

                print("=" * 70)

            # -------- SAVE -------- #
            if update_count % self.save_interval == 0:
                self.save_checkpoint(update_count, timesteps)

        print("\nTraining completed!")
        self.save_checkpoint(update_count, timesteps, final=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    def save_checkpoint(self, update_count, timesteps, final=False):

        import os
        os.makedirs(self.save_dir, exist_ok=True)

        path = (
            f"{self.save_dir}/final_model.pt"
            if final else
            f"{self.save_dir}/checkpoint_{update_count}.pt"
        )

        self.mappo.save(path)

        print(f"✓ Saved model -> {path}")

    # --------------------------------------------------------
    # Evaluation + Video
    # --------------------------------------------------------

    def evaluate(self, n_episodes=5, render=True):

        print("\n" + "=" * 70)
        print("Evaluating & Recording")
        print("=" * 70)

        eval_env = WarehouseEnv(
            height=self.env.height,
            width=self.env.width,
            n_robots=self.env.n_robots,
            max_steps=self.env.max_steps,
            render_mode="rgb_array" if render else None
        )

        recorder = None
        if render:
            try:
                from video_recorder import VideoRecorder
                import os
                os.makedirs("videos", exist_ok=True)
                recorder = VideoRecorder("videos/mappo_eval.mp4", fps=10)
                print("🎥 Recording to videos/mappo_eval.mp4")
            except Exception as e:
                print(f"⚠ Could not init recorder: {e}")

        rewards_list = []

        for ep in range(n_episodes):

            obs, _ = eval_env.reset()
            done = False
            ep_reward = 0
            step = 0

            while not done:

                global_state = eval_env.get_global_state()
                actions = {}

                for agent in eval_env.agents:
                    agent_id = int(agent.split("_")[1])

                    action, _, _ = self.mappo.select_action(
                        obs[agent],
                        global_state,
                        agent_id,
                        deterministic=True
                    )
                    actions[agent] = action

                obs, rewards, terms, truncs, _ = eval_env.step(actions)

                if render and recorder:
                    frame = eval_env.render()
                    if frame is not None:
                        recorder.add_frame(frame)

                ep_reward += np.mean(list(rewards.values()))
                done = any(terms.values()) or any(truncs.values())
                step += 1

            rewards_list.append(ep_reward)
            print(f"Episode {ep+1} | Reward: {ep_reward:.2f} | Steps: {step}")

        if recorder:
            recorder.close()
            print("✓ Video saved!")

        eval_env.close()

        print("\nRESULTS")
        print(f"Mean reward: {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
        print("=" * 70)
