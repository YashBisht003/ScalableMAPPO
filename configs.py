MASTER_SEED = 42
FIXED_LAYOUT = True

CONFIG = {
    # Environment - SCALED UP
    'height': 17,
    'width': 17,
    'n_robots': 10,  # Doubled!
    'n_shelves': 16,  # More shelves
    'n_drops': 4,  # More drop points
    'max_steps': 800,  # Longer episodes
    'view_size': 5,

    # MAPPO - Tuned for larger scale
    'lr_actor': 2e-4,  # Slightly lower for stability
    'lr_critic': 8e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.015,  # Slightly higher for exploration
    'value_coef': 0.5,

    # Training - More samples needed
    'total_timesteps': 2_000_000,  # Train longer
    'n_steps': 4096,  # Larger rollout buffer
    'n_epochs': 4,
    'batch_size': 128,  # Larger batches
    'save_interval': 50,
    'log_interval': 10,

    # New features
    'battery_enabled': True,
    'task_system_enabled': True,
    'communication_enabled': True,
}