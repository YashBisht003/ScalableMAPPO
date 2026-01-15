from WarehouseEnv import WarehouseEnv
from scalablemappo import ScalableMAPPO
from trainer import MAPPOTrainer
from configs import CONFIG, MASTER_SEED

import numpy as np
import torch

# ---------------- SEED ---------------- #
np.random.seed(MASTER_SEED)
torch.manual_seed(MASTER_SEED)


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
        n_actions=8,                         # <-- CHARGE action
        n_agents=CONFIG['n_robots'],        # <-- IMPORTANT
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
    )

    # -------- TRAIN -------- #
    trainer.train(total_timesteps=CONFIG['total_timesteps'])

    # -------- EVALUATE -------- #
    env.render_mode = "human"
    trainer.evaluate(n_episodes=5, render=True)


if __name__ == "__main__":
    main()
