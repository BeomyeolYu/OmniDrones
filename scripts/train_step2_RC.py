import os
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
from plot_trajectories import plot_traj_collect

# #######################################################################################
"""
'''
Save Test Data
'''
import hydra
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
)
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from tensordict import TensorDict

# ==================================
# Configuration
# ==================================
inverted_pendulum = True

s2r_gap = 130 #100
dt = "0p016" #"0p005"
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 1000 #500 # The target length of the collected data
SAVE_CHUNK_SIZE = 100 # Save data every 100 episodes
BASE_SAVE_NAME = (  # New: Base name for chunked files
    f"deltaTrainingDataset_{SAVE_CHUNK_SIZE}trajs_{MAX_STEPS_PER_EPISODE}steps"
)
if inverted_pendulum:
    BASE_SAVE_NAME = (  # New: Base name for chunked files
        f"deltaTrainingDataset_Inv_{SAVE_CHUNK_SIZE}trajs_{MAX_STEPS_PER_EPISODE}steps"
    )
# ==================================

@hydra.main(version_base=None, config_path=".", config_name="train_f450")
def main(cfg):
    # --- Environment and Policy Setup ---
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

    # --- Data Collection ---
    all_states, all_next_states, all_actions, all_dones = [], [], [], []

    ep = 0
    pbar = tqdm(total=NUM_EPISODES, desc="Collecting Trajectories")
    while ep < NUM_EPISODES:
        obs = env.reset()

        # Warm-up step to discard the initial state and align data tuples
        act_td = TensorDict({("agents", "action"): env.action_spec.zero()}, batch_size=[1])
        obs = env.step(act_td)["next"]

        steps = 0
        traj_states, traj_next_states, traj_actions, traj_dones = [], [], [], []
        
        crashed = False # Initialize crashed status for the episode
        
        # Main data collection loop for a single trajectory
        while steps < MAX_STEPS_PER_EPISODE:
            error_obs = obs[("agents", "observation")].to(base_env.device)
            obs_td = TensorDict({("agents", "observation"): error_obs}, batch_size=[1], device=base_env.device)

            with torch.no_grad():
                act = policy(obs_td)[("agents", "action")]

            state = base_env.get_state()
            act_td = TensorDict({("agents", "action"): act}, batch_size=[1], device=base_env.device)
            next_obs = env.step(act_td)
            next_state = base_env.get_state()

            # The 'terminated' flag indicates a crash or failure state
            crashed = next_obs["next"]["terminated"].item()
            # The episode is 'done' if it crashed OR it reached the last step
            done = crashed or (steps == MAX_STEPS_PER_EPISODE - 1)

            traj_states.append(state)
            traj_next_states.append(next_state)
            traj_actions.append(act.cpu().numpy().flatten())
            traj_dones.append(int(done))

            obs = next_obs["next"]
            steps += 1
            
            # Exit the inner loop if the episode is done (crashed or finished)
            if done:
                break
        
        # =================================================================
        # MODIFIED: Trajectory Saving/Discarding Logic
        # =================================================================
        traj_len = len(traj_states)
        # We only save the trajectory if it ran for the FULL duration AND did NOT crash.
        if traj_len == MAX_STEPS_PER_EPISODE and not crashed:
            all_states.append(np.array(traj_states))
            all_next_states.append(np.array(traj_next_states))
            all_actions.append(np.array(traj_actions))
            all_dones.append(np.array(traj_dones))
            ep += 1 # Only increment the saved episode count on success
            pbar.update(1)
            pbar.set_postfix({"last_traj_len": traj_len, "crashed": crashed})
        else:
            # If it's too short, it must have crashed. Discard and retry.
            print(f"Discarded trajectory with {traj_len} steps (crashed={crashed}). Retrying...")
        # =================================================================

        # =================================================================
        # CHECKPOINTING LOGIC (No changes needed here)
        # =================================================================
        is_last_episode = ep == NUM_EPISODES
        # Save if the chunk is full AND if there's data to save
        if (ep > 0 and ep % SAVE_CHUNK_SIZE == 0) or is_last_episode:
            if not all_states: # Skip if the buffer is empty
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_filename = f"{BASE_SAVE_NAME}_chunk_{timestamp}.npz"
            
            print(f"\nCheckpointing: Saving {len(all_states)} trajectories to {chunk_filename}...")
            
            np.savez(
                chunk_filename,
                states=np.array(all_states, dtype=object),
                next_states=np.array(all_next_states, dtype=object),
                actions=np.array(all_actions, dtype=object),
                dones=np.array(all_dones, dtype=object),
            )
            print("Save complete.")

            # Clear the lists to free up memory for the next chunk
            all_states.clear()
            all_next_states.clear()
            all_actions.clear()
            all_dones.clear()
        # =================================================================

    simulation_app.close()


if __name__ == "__main__":
    main()
"""

#############################################################################
# import os
# import hydra
# import torch
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# from omegaconf import OmegaConf
# from tqdm import tqdm

# from omni_drones import init_simulation_app
# from omni_drones.learning import ALGOS
# from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
# from tensordict import TensorDict

# # The visualization function `plot_traj_collect` is unchanged.
# # ==============================================================================
# def plot_traj_collect(npz_file, num_to_plot=5):
#     # ... (code for plotting is unchanged)
#     pass
# # ==============================================================================

# @hydra.main(version_base=None, config_path=".", config_name="train_f450")
# def main(cfg):
#     # --- 1. Environment and Policy Setup ---
#     OmegaConf.resolve(cfg); OmegaConf.set_struct(cfg, False)
#     simulation_app = init_simulation_app(cfg)
    
#     from omni_drones.envs.isaac_env import IsaacEnv
#     env_class = IsaacEnv.REGISTRY[cfg.task.name]
#     base_env = env_class(cfg, headless=True)
#     env = TransformedEnv(base_env, Compose(InitTracker())).train()
    
#     policy = ALGOS[cfg.algo.name.lower()](
#         cfg.algo, env.observation_spec, env.action_spec, 
#         env.reward_spec, device=base_env.device
#     )
    
#     if os.path.exists(cfg.collection.pretrained_ckpt_path):
#         print(f"Loading pre-trained policy from {cfg.collection.pretrained_ckpt_path}")
#         policy.load_state_dict(torch.load(cfg.collection.pretrained_ckpt_path, weights_only=True))
#     else:
#         raise FileNotFoundError(f"Pre-trained policy not found at {cfg.collection.pretrained_ckpt_path}")
#     policy.eval()

#     # --- 2. Data Collection Setup ---
#     output_dir = os.getcwd()
#     data_save_dir = os.path.join(output_dir, "deltaAction_trainingDataset")
#     os.makedirs(data_save_dir, exist_ok=True)
#     print(f"Data chunks will be saved in: {data_save_dir}")
    
#     trajectories_for_chunk = []
#     base_save_name = (
#         f"deltaTrainingDataset_{cfg.task.env.num_envs}envs_{cfg.collection.save_chunk_size}trajs"
#     )
#     pbar = tqdm(total=cfg.collection.num_trajectories_to_collect, desc="Collecting Trajectories")

#     # --- 3. Main Collection Loop (MANUAL ROLLOUT WITH WARM-UP) ---
#     while pbar.n < cfg.collection.num_trajectories_to_collect:
        
#         # Manually perform a multi-environment rollout
#         td = env.reset()

#         # === THE DEFINITIVE FIX: Perform a warm-up step ===
#         action_dim = base_env.drone.action_spec.shape[-1]
#         zero_actions_tensor = torch.zeros(
#             base_env.num_envs, 1, action_dim, 
#             device=base_env.device
#         )
#         zero_actions_td = TensorDict(
#             {("agents", "action"): zero_actions_tensor},
#             batch_size=base_env.num_envs
#         )
#         td = env.step(zero_actions_td)
#         # =========================================================
        
#         # Buffers to store the data for this batch of rollouts
#         batch_states, batch_actions, batch_next_states, batch_dones = [], [], [], []
#         is_done = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
        
#         with torch.no_grad():
#             for _ in range(base_env.max_episode_length):
#                 # Mask out actions for environments that are already done
#                 current_obs_td = td["next"].clone()
#                 action_td = policy(current_obs_td)
#                 action_td[("agents", "action")][is_done] = 0. # Set action to zero for done envs

#                 # Append current state and action to buffers
#                 batch_states.append(current_obs_td["state"])
#                 batch_actions.append(action_td[("agents", "action")])
                
#                 td = env.step(action_td)
                
#                 batch_next_states.append(td[("next", "state")])
#                 batch_dones.append(td[("next", "done")])

#                 # Update the done mask
#                 is_done = is_done | td[("next", "done")].squeeze(-1)
#                 if is_done.all(): # Stop if all environments are done
#                     break

#         # Stack the collected tensors
#         data_batch = TensorDict({
#             "state": torch.stack(batch_states),
#             ("agents", "action"): torch.stack(batch_actions),
#             "next": {
#                 "state": torch.stack(batch_next_states),
#                 "done": torch.stack(batch_dones)
#             }
#         }, batch_size=[len(batch_states), base_env.num_envs])

#         data_batch = data_batch.permute(1, 0)
#         done_indices = torch.where(data_batch[("next", "done")].squeeze(-1))
        
#         # # --- 4. Process and Save Trajectories ---
#         # for i in range(base_env.num_envs):
#         #     if pbar.n >= cfg.collection.num_trajectories_to_collect: break
            
#         #     ends = torch.where(done_indices[0] == i)[0]
#         #     if not len(ends): continue
            
#         #     start = 0
#         #     for end_idx in ends:
#         #         end_step = done_indices[1][end_idx].item()
#         #         if pbar.n >= cfg.collection.num_trajectories_to_collect: break
                
#         #         traj_td = data_batch[i, start : end_step + 1]
                
#         #         if len(traj_td) >= cfg.collection.min_traj_len:
#         #             trajectories_for_chunk.append({
#         #                 "states": traj_td["state"].cpu().numpy(),
#         #                 "actions": traj_td[("agents", "action")].squeeze(1).cpu().numpy(),
#         #                 "next_states": traj_td[("next", "state")].cpu().numpy(),
#         #                 "dones": traj_td[("next", "done")].squeeze(-1).cpu().numpy(),
#         #             })
#         #             pbar.update(1)
#         #         start = end_step + 1

#         # --- 4. Process and Save Trajectories (Corrected Logic) ---
#         for i in range(base_env.num_envs):
#             if pbar.n >= cfg.collection.num_trajectories_to_collect: break

#             # THE FIX: Access the data by key FIRST, then slice by index.
#             done_tensor_for_env = data_batch["next", "done"][i].squeeze(-1)
#             done_steps_for_env = torch.where(done_tensor_for_env)[0]
            
#             # Sort the termination steps chronologically.
#             sorted_done_steps = torch.sort(done_steps_for_env)[0]
            
#             start_step = 0
#             # Iterate through the correctly ordered crash/truncation points
#             for end_step_idx in sorted_done_steps:
#                 end_step = end_step_idx.item() + 1
                
#                 # Slice the trajectory from the last "done" to the current one
#                 traj_td = data_batch[i, start_step:end_step]
                
#                 if len(traj_td) >= cfg.collection.min_traj_len:
#                     if pbar.n >= cfg.collection.num_trajectories_to_collect: break
                    
#                     trajectories_for_chunk.append({
#                         "states": traj_td["state"].cpu().numpy(),
#                         "actions": traj_td[("agents", "action")].squeeze(1).cpu().numpy(),
#                         "next_states": traj_td[("next", "state")].cpu().numpy(),
#                         "dones": traj_td[("next", "done")].squeeze(-1).cpu().numpy(),
#                     })
#                     pbar.update(1)
            
#                 # Update the start for the next trajectory slice
#                 start_step = end_step

#         # --- 5. Checkpointing Logic ---
#         if len(trajectories_for_chunk) >= cfg.collection.save_chunk_size:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             chunk_filename = os.path.join(data_save_dir, f"{base_save_name}_chunk_{timestamp}.npz")
            
#             print(f"\nCheckpointing: Saving {len(trajectories_for_chunk)} trajectories to {chunk_filename}...")
#             np.savez(
#                 chunk_filename,
#                 states=np.array([t["states"] for t in trajectories_for_chunk], dtype=object),
#                 actions=np.array([t["actions"] for t in trajectories_for_chunk], dtype=object),
#                 next_states=np.array([t["next_states"] for t in trajectories_for_chunk], dtype=object),
#                 dones=np.array([t["dones"] for t in trajectories_for_chunk], dtype=object),
#             )
#             print("Save complete.")
#             trajectories_for_chunk.clear()

#     # --- Final Save ---
#     if trajectories_for_chunk:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         chunk_filename = os.path.join(data_save_dir, f"{base_save_name}_chunk_{timestamp}.npz")
#         print(f"\nSaving final {len(trajectories_for_chunk)} trajectories to {chunk_filename}...")
#         np.savez(
#             chunk_filename,
#             states=np.array([t["states"] for t in trajectories_for_chunk], dtype=object),
#             actions=np.array([t["actions"] for t in trajectories_for_chunk], dtype=object),
#             next_states=np.array([t["next_states"] for t in trajectories_for_chunk], dtype=object),
#             dones=np.array([t["dones"] for t in trajectories_for_chunk], dtype=object),
#         )

#     simulation_app.close()
#     pbar.close()
#     print("\nData collection complete.")

# if __name__ == "__main__":
#     main()

#############################################################################
import os
import hydra
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from tensordict import TensorDict

# ==============================================================================
# ==== 1. PLOTTING & SUMMARY FUNCTIONS (Adapted for Final Deployment) =========
# ==============================================================================

def plot_traj_collect(npz_file, num_to_plot=5):
    # ... (code for plotting is unchanged)
    pass
# ==============================================================================

@hydra.main(version_base=None, config_path=".", config_name="train_f450")
def main(cfg):
    # --- 1. Environment and Policy Setup ---
    OmegaConf.resolve(cfg); OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    
    from omni_drones.envs.isaac_env import IsaacEnv
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=True)
    env = TransformedEnv(base_env, Compose(InitTracker())).train()
    
    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec, 
        env.reward_spec, device=base_env.device
    )
    
    if os.path.exists(cfg.collection.pretrained_ckpt_path):
        print(f"Loading pre-trained policy from {cfg.collection.pretrained_ckpt_path}")
        policy.load_state_dict(torch.load(cfg.collection.pretrained_ckpt_path, weights_only=True))
    else:
        raise FileNotFoundError(f"Pre-trained policy not found at {cfg.collection.pretrained_ckpt_path}")
    policy.eval()

    # --- 2. Data Collection Setup ---
    output_dir = os.getcwd()
    data_save_dir = os.path.join(output_dir, "deltaAction_trainingDataset")
    os.makedirs(data_save_dir, exist_ok=True)
    print(f"Data chunks will be saved in: {data_save_dir}")
    
    trajectories_for_chunk = []
    base_save_name = (
        f"deltaTrainingDataset_{cfg.task.env.num_envs}envs_{cfg.collection.save_chunk_size}trajs"
    )
    pbar = tqdm(total=cfg.collection.num_trajectories_to_collect, desc="Collecting Trajectories")

    # --- 3. Main Collection Loop (MANUAL ROLLOUT WITH WARM-UP) ---
    while pbar.n < cfg.collection.num_trajectories_to_collect:
        
        # Manually perform a multi-environment rollout
        td = env.reset()

        # === THE DEFINITIVE FIX: Perform a warm-up step ===
        action_dim = base_env.drone.action_spec.shape[-1]
        zero_actions_tensor = torch.zeros(
            base_env.num_envs, 1, action_dim, 
            device=base_env.device
        )
        zero_actions_td = TensorDict(
            {("agents", "action"): zero_actions_tensor},
            batch_size=base_env.num_envs
        )
        td = env.step(zero_actions_td)
        # =========================================================
        
        # Buffers to store the data for this batch of rollouts
        batch_states, batch_actions, batch_next_states, batch_dones = [], [], [], []
        is_done = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
        
        with torch.no_grad():
            """
            for _ in range(base_env.max_episode_length):
                # Mask out actions for environments that are already done
                current_obs_td = td["next"].clone()
                action_td = policy(current_obs_td)
                action_td[("agents", "action")][is_done] = 0. # Set action to zero for done envs

                # Append current state and action to buffers
                batch_states.append(current_obs_td["state"])
                batch_actions.append(action_td[("agents", "action")])
                
                td = env.step(action_td)
            """
            # Get the noise level from your config.
            # Add 'exploration_noise' to your hydra config file.
            noise_level = cfg.collection.get("exploration_noise", 0.05)
            if noise_level > 0.0:
                print(f"Using exploration noise level: {noise_level}")
                # Set policy to 'eval' but allow for noise addition
                policy.eval() 
            else:
                policy.eval()

            for _ in range(base_env.max_episode_length):
                # Mask out actions for environments that are already done
                current_obs_td = td["next"].clone()
                action_td = policy(current_obs_td)
                # --- ADD EXPLORATION NOISE ---
                base_action = action_td[("agents", "action")]
                # Generate Gaussian noise (scaled by noise_level)
                noise = torch.randn_like(base_action) * noise_level

                # Add noise and clamp the result to the valid action range [-1, 1]
                final_action = torch.clamp(base_action + noise, min=-1.0, max=1.0)
                
                # Apply the original done mask *after* adding noise
                final_action[is_done] = 0.0

                # Put the final noisy action back into the TensorDict
                action_td[("agents", "action")] = final_action
                # --- END NOISE ADDITION ---

                # Append current state and *final* action to buffers
                batch_states.append(current_obs_td["state"])
                batch_actions.append(action_td[("agents", "action")]) # This now appends the noisy action
                td = env.step(action_td) # This now steps with the noisy action
                
                batch_next_states.append(td[("next", "state")])
                batch_dones.append(td[("next", "done")])

                # Update the done mask
                is_done = is_done | td[("next", "done")].squeeze(-1)
                if is_done.all(): # Stop if all environments are done
                    break

        # Stack the collected tensors
        data_batch = TensorDict({
            "state": torch.stack(batch_states),
            ("agents", "action"): torch.stack(batch_actions),
            "next": {
                "state": torch.stack(batch_next_states),
                "done": torch.stack(batch_dones)
            }
        }, batch_size=[len(batch_states), base_env.num_envs])

        data_batch = data_batch.permute(1, 0) # [NumEnvs, T, ...]
        
        # --- 4. Process and Save Trajectories (Corrected Logic) ---
        for i in range(base_env.num_envs):
            if pbar.n >= cfg.collection.num_trajectories_to_collect: break

            # Access the data by key FIRST, then slice by index.
            done_tensor_for_env = data_batch["next", "done"][i].squeeze(-1)
            done_steps_for_env = torch.where(done_tensor_for_env)[0]
            
            # --- Find the *first* time the environment was done.
            if len(done_steps_for_env) > 0:
                # Get the first termination index
                first_done_step = torch.min(done_steps_for_env).item()
                end_step = first_done_step + 1
            else:
                # The episode never terminated (ran for full length)
                end_step = base_env.max_episode_length
                # Ensure we don't slice past the collected data
                end_step = min(end_step, data_batch.shape[1]) 
            
            # Slice the *one and only* trajectory from start to first termination.
            traj_td = data_batch[i, 0:end_step]
                
            if len(traj_td) >= cfg.collection.min_traj_len:
                
                trajectories_for_chunk.append({
                    "states": traj_td["state"].cpu().numpy(),
                    "actions": traj_td[("agents", "action")].squeeze(1).cpu().numpy(),
                    "next_states": traj_td[("next", "state")].cpu().numpy(),
                    "dones": traj_td[("next", "done")].squeeze(-1).cpu().numpy(),
                })
                pbar.update(1)

                # --- THIS IS THE FIX ---
                # Checkpointing logic is moved *inside* the loop.
                # It checks after *every* trajectory is added.
                if len(trajectories_for_chunk) >= cfg.collection.save_chunk_size:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    chunk_filename = os.path.join(data_save_dir, f"{base_save_name}_chunk_{timestamp}.npz")
                    
                    print(f"\nCheckpointing: Saving {len(trajectories_for_chunk)} trajectories to {chunk_filename}...")
                    np.savez(
                        chunk_filename,
                        states=np.array([t["states"] for t in trajectories_for_chunk], dtype=object),
                        actions=np.array([t["actions"] for t in trajectories_for_chunk], dtype=object),
                        next_states=np.array([t["next_states"] for t in trajectories_for_chunk], dtype=object),
                        dones=np.array([t["dones"] for t in trajectories_for_chunk], dtype=object),
                    )
                    print("Save complete.")
                    trajectories_for_chunk.clear()
                # --- END FIX ---

        # --- 5. Checkpointing Logic (This is now redundant but harmless) ---
        # This check is no longer strictly necessary, but we leave it
        # just in case the loop finishes before the pbar limit.
        if len(trajectories_for_chunk) >= cfg.collection.save_chunk_size:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_filename = os.path.join(data_save_dir, f"{base_save_name}_chunk_{timestamp}.npz")
            
            print(f"\nCheckpointing (end of batch): Saving {len(trajectories_for_chunk)} trajectories to {chunk_filename}...")
            np.savez(
                chunk_filename,
                states=np.array([t["states"] for t in trajectories_for_chunk], dtype=object),
                actions=np.array([t["actions"] for t in trajectories_for_chunk], dtype=object),
                next_states=np.array([t["next_states"] for t in trajectories_for_chunk], dtype=object),
                dones=np.array([t["dones"] for t in trajectories_for_chunk], dtype=object),
            )
            print("Save complete.")
            trajectories_for_chunk.clear()

    # --- Final Save ---
    if trajectories_for_chunk:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunk_filename = os.path.join(data_save_dir, f"{base_save_name}_chunk_{timestamp}.npz")
        print(f"\nSaving final {len(trajectories_for_chunk)} trajectories to {chunk_filename}...")
        np.savez(
            chunk_filename,
            states=np.array([t["states"] for t in trajectories_for_chunk], dtype=object),
            actions=np.array([t["actions"] for t in trajectories_for_chunk], dtype=object),
            next_states=np.array([t["next_states"] for t in trajectories_for_chunk], dtype=object),
            dones=np.array([t["dones"] for t in trajectories_for_chunk], dtype=object),
        )

    simulation_app.close()
    pbar.close()
    print("\nData collection complete.")

if __name__ == "__main__":
    main()