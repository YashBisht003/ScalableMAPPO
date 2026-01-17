import os
from WarehouseEnv import WarehouseEnv
from scalablemappo import ScalableMAPPO
from trainer import MAPPOTrainer
from configs import CONFIG, MASTER_SEED, SAVE_DIR

import numpy as np
import torch

# ---------------- SEED ---------------- #
np.random.seed(MASTER_SEED)
torch.manual_seed(MASTER_SEED)


# --------------------------------------------------
# Find latest checkpoint automatically
# --------------------------------------------------
def find_latest_checkpoint():
    if not os.path.exists(SAVE_DIR):
        return None

    files = [f for f in os.listdir(SAVE_DIR) if f.startswith("checkpoint_")]
    if not files:
        return None

    return max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))


def main():

    print("\nCreating environment...")
    env = WarehouseEnv(
        height=CONFIG['height'],
        width=CONFIG['width'],
        n_robots=CONFIG['n_robots'],
        max_steps=CONFIG['max_steps'],
        view_size=CONFIG['view_size'],
        render_mode=None   # change to "human" for debug
    )

    # -------- SAMPLE SHAPES -------- #
    obs_sample = env.reset()[0][env.agents[0]]
    global_state_sample = env.get_global_state()

    print("Creating Scalable MAPPO agent...")
    mappo = ScalableMAPPO(
        obs_channels=obs_sample.shape[0],
        view_size=obs_sample.shape[1],
        n_actions=8,                         
        n_agents=CONFIG['n_robots'],        
        global_state_dim=len(global_state_sample),
        lr_actor=CONFIG['lr_actor'],
        lr_critic=CONFIG['lr_critic'],
        gamma=CONFIG['gamma'],
        gae_lambda=CONFIG['gae_lambda'],
        clip_epsilon=CONFIG['clip_epsilon'],
        entropy_coef=CONFIG['entropy_coef'],
        value_coef=CONFIG['value_coef'],
    )

    print("Creating trainer...")
    trainer = MAPPOTrainer(
        env=env,
        mappo=mappo,
        n_steps=CONFIG['n_steps'],
        n_epochs=CONFIG['n_epochs'],
        batch_size=CONFIG['batch_size'],
        save_interval=CONFIG['save_interval'],
        log_interval=CONFIG['log_interval'],
        save_dir=SAVE_DIR
    )

    # =====================================================
    # 🔥 AUTO RESUME LATEST CHECKPOINT
    # =====================================================
    ckpt = find_latest_checkpoint()

    if ckpt:
        RESUME_PATH = f"{SAVE_DIR}/{ckpt}"
        print("=" * 60)
        print(f"Resuming training from: {RESUME_PATH}")
        mappo.load(RESUME_PATH)
        print("Checkpoint loaded successfully!")
        print("=" * 60)
    else:
        print("No checkpoint found, training from scratch.")

    # -------- TRAIN -------- #
    trainer.train(total_timesteps=CONFIG['total_timesteps'])

    # -------- EVALUATE -------- #
    env.render_mode = "human"
    trainer.evaluate(n_episodes=5, render=True)


if __name__ == "__main__":
    main()
