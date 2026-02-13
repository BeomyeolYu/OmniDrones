import os
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
from plot_trajectories import plot_traj_collect

import hydra
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from omni_drones import init_simulation_app
from omni_drones.utils.wandb import init_wandb
from omni_drones.learning import ALGOS
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
)
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from tensordict import TensorDict


s2r_gap = 120
dt = "0p005" #"0p016"
NUM_EPISODES = 3  #100
MAX_STEPS = 500 + 1 # 1000 + 1 # +1 because we will discard the first step
MIN_TRAJ_LEN = 100 # Minimum trajectory length to save, even if it's a failure
SAVE_FILENAME = "deltaTrainingDataset_"+str(NUM_EPISODES)+"trajs_"+str(MAX_STEPS - 1)+"steps_"+dt+"dt_"+str(s2r_gap)+"gaps.npz"

margins = .95
CRASH_THRESHOLD = 0.1  # if y3 (state[2]) < this → treat as crash
Y_LIM = 1.0*margins # out of boundary [m]
Y_DOT_LIM = 4.0*margins # [m/s]
w_LIM = W_LIM = 2*torch.pi*margins # [rad/s]

@hydra.main(version_base=None, config_path=".", config_name="train_f450")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    transforms = [InitTracker()]
    if cfg.task.get("ravel_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("ravel_obs_central", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation_central")))

    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromMultiDiscreteAction(nbins=nbins))
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromDiscreteAction(nbins=nbins))
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device,
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_ckpt_name = cfg.get("pretrained_ckpt_path", None)
    if pretrained_ckpt_name:
        pretrained_ckpt_path = os.path.join(base_dir, pretrained_ckpt_name)
        if os.path.exists(pretrained_ckpt_path):
            policy.load_state_dict(torch.load(pretrained_ckpt_path, map_location=base_env.device, weights_only=True))

    all_states, all_next_states, all_actions, all_dones = [], [], [], []

    ep = 0
    pbar = tqdm(total=NUM_EPISODES, desc="Collecting Trajectories")
    while ep < NUM_EPISODES:
        obs = env.reset()
        steps = 0

        traj_states, traj_next_states, traj_actions, traj_dones = [], [], [], []
        
        # --- Main data collection loop for a single trajectory ---
        while steps < MAX_STEPS:
            error_obs = obs[("agents", "observation")].to(base_env.device)
            obs_td = TensorDict({("agents", "observation"): error_obs}, batch_size=[1], device=base_env.device)

            with torch.no_grad():
                act = policy(obs_td)[("agents", "action")]

            state = base_env.get_state()
            act_td = TensorDict({("agents", "action"): act}, batch_size=[1], device=base_env.device)
            next_obs = env.step(act_td)
            next_state = base_env.get_state()

            # --- Crash and episode termination detection ---
            '''
            next_state_tensor = torch.from_numpy(next_state).to(base_env.device)
            crashed = (
                torch.any(torch.abs(next_state_tensor[0:2]) > Y_LIM)
                or next_state_tensor[2] < CRASH_THRESHOLD
                or torch.any(torch.abs(next_state_tensor[3:6]) > Y_DOT_LIM)
                or torch.any(torch.abs(next_state_tensor[9:12]) > w_LIM)
                or torch.any(torch.abs(next_state_tensor[21:]) > W_LIM)
            )
            '''
            crashed = next_obs["next"]["terminated"].item() # Get the ground truth directly from the environment

            # The episode is 'done' if it crashed or reached the maximum step count
            done = crashed or (steps == MAX_STEPS - 1)

            # Add this line immediately after
            if not crashed and (steps == MAX_STEPS - 1):
                crashed = False # Explicitly mark successful timeouts as not crashed

            # Discard the very first step of each trajectory
            if steps != 0:
                traj_states.append(state)
                traj_next_states.append(next_state)
                traj_actions.append(act.cpu().numpy().flatten())
                traj_dones.append(int(done))

            obs = next_obs["next"]
            steps += 1
            
            # If the episode is finished, break the loop to evaluate the trajectory
            if done:
                break
        
        # --- Trajectory Saving/Discarding Logic ---
        # The actual length of the collected trajectory data
        traj_len = len(traj_states)

        # Only save trajectories (successful or failed) that are long enough
        if traj_len >= MIN_TRAJ_LEN:
            all_states.append(np.array(traj_states))
            all_next_states.append(np.array(traj_next_states))
            all_actions.append(np.array(traj_actions))
            all_dones.append(np.array(traj_dones))
            ep += 1
            pbar.update(1)
            pbar.set_postfix({"last_traj_len": traj_len, "crashed": crashed})
        else:
            # This handles both crashes at the very beginning and other short, invalid trajectories
            print(f"Discarded trajectory with {traj_len} steps (crashed={crashed}). Retrying...")

    # Save the lists directly as object arrays
    np.savez(
        SAVE_FILENAME,
        states=np.array(all_states, dtype=object),
        next_states=np.array(all_next_states, dtype=object),
        actions=np.array(all_actions, dtype=object),
        dones=np.array(all_dones, dtype=object),
    )
    print(f"\nSaved padded dataset to {SAVE_FILENAME}")

    plot_traj_collect(SAVE_FILENAME)

    simulation_app.close()

if __name__ == "__main__":
    main()

#######################################################################################
"""
# --- 1. Define Parameters & Filenames ---
# The file saved by your collection script
INPUT_FILENAME = "delta_training_data/delta_training_data_500trajs_1000steps_0p005dt_120gaps.npz"
# The new file we will create with cleaned data
OUTPUT_FILENAME = "delta_training_data/cleaned_delta_training_data_500trajs_1000steps_0p005dt_120gaps.npz"

# Define the state boundaries exactly as in your script
CRASH_THRESHOLD = 0.1  # Minimum altitude for state[2]
Y_LIM = 1.0            # Max absolute value for position state[0] and state[1]
Y_DOT_LIM = 4.0        # Max absolute value for linear velocity state[3:6]
W_LIM = 2 * np.pi      # Max absolute value for angular velocity state[9:12] and rotor speeds state[21:]

# --- 2. Load the Data ---
try:
    data = np.load(INPUT_FILENAME)
    print(f"Successfully loaded data from '{INPUT_FILENAME}'")
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILENAME}' was not found. Please check the path.")
    exit()

# Extract the arrays
# Shape is (num_trajectories, steps_per_trajectory, state_dimension)
all_states = data['states']
all_next_states = data['next_states']
all_actions = data['actions']
all_dones = data['dones']

original_traj_count = all_states.shape[0]
print(f"Original dataset contains {original_traj_count} trajectories.")

# --- 3. Find Invalid Trajectories using Vectorized Checks ---
# We check every state in every trajectory simultaneously.

# State structure: [y(3), y_dot(3), q(3), w(3), R(9), W(3)]
# Indices:
# y: 0, 1, 2
# y_dot: 3, 4, 5
# w: 9, 10, 11
# W: 21, 22, 23

# Check for violations across all timesteps for each trajectory.
# The result of each check is a boolean array of shape (num_trajectories,).

# Condition 1: Horizontal position exceeds Y_LIM
invalid_pos_xy = np.any(np.abs(all_states[:, :, 0:2]) > Y_LIM, axis=(1, 2)) # CHANGED

# Condition 2: Altitude drops below CRASH_THRESHOLD (This one was correct)
invalid_pos_z = np.any(all_states[:, :, 2] < CRASH_THRESHOLD, axis=1)

# Condition 3: Linear velocity exceeds Y_DOT_LIM
invalid_vel_linear = np.any(np.abs(all_states[:, :, 3:6]) > Y_DOT_LIM, axis=(1, 2)) # CHANGED

# Condition 4: Angular velocity exceeds W_LIM
invalid_vel_angular = np.any(np.abs(all_states[:, :, 9:12]) > W_LIM, axis=(1, 2)) # CHANGED

# Condition 5: Rotor speeds exceed W_LIM
invalid_rotor_speed = np.any(np.abs(all_states[:, :, 21:]) > W_LIM, axis=(1, 2)) # CHANGED

# Combine all invalid conditions. A trajectory is invalid if ANY condition is true.
is_invalid = (
    invalid_pos_xy |
    invalid_pos_z |
    invalid_vel_linear |
    invalid_vel_angular |
    invalid_rotor_speed
)

# --- 4. Filter the Data ---
# Get the indices of the trajectories that are NOT invalid.
valid_indices = ~is_invalid

# Use these boolean indices to select only the valid trajectories
cleaned_states = all_states[valid_indices]
cleaned_next_states = all_next_states[valid_indices]
cleaned_actions = all_actions[valid_indices]
cleaned_dones = all_dones[valid_indices]

new_traj_count = cleaned_states.shape[0]
print(f"\nFiltering complete.")
print(f"Removed {original_traj_count - new_traj_count} invalid trajectories.")
print(f"New dataset contains {new_traj_count} valid trajectories.")

# --- 5. Save the Cleaned Data ---
np.savez(
    OUTPUT_FILENAME,
    states=cleaned_states,
    next_states=cleaned_next_states,
    actions=cleaned_actions,
    dones=cleaned_dones,
)

print(f"\nSuccessfully saved cleaned data to '{OUTPUT_FILENAME}'")
plot_traj_collect(OUTPUT_FILENAME)
"""