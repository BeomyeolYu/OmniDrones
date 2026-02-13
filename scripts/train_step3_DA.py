# import logging
# import os
# import gc
# import glob
# import time

# import hydra
# import torch
# import numpy as np
# import pandas as pd
# import wandb
# import matplotlib.pyplot as plt

# from torch.func import vmap
# from tqdm import tqdm
# from omegaconf import OmegaConf

# from omni_drones import init_simulation_app
# from torchrl.data import CompositeSpec
# from torchrl.envs.utils import set_exploration_type, ExplorationType
# from omni_drones.utils.torchrl import SyncDataCollector
# from omni_drones.utils.torchrl.transforms import (
#     FromMultiDiscreteAction,
#     FromDiscreteAction,
#     ravel_composite,
#     AttitudeController,
#     RateController,
# )
# from omni_drones.utils.wandb import init_wandb
# from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
# from omni_drones.learning import ALGOS

# from setproctitle import setproctitle
# from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
# from torchrl.envs.transforms import CatFrames 

# def load_and_merge_chunks(directory_path: str):
#     """
#     Finds all .npz chunks in a directory, merges them into a single 
#     dataset in memory, and returns the data as a dictionary. This version
#     is optimized for lower memory usage and handles different numpy array formats.
#     """
#     if not os.path.isdir(directory_path):
#         raise FileNotFoundError(f"Provided data path is not a directory: {directory_path}")

#     chunk_pattern = os.path.join(directory_path, "*_chunk_*.npz")
#     chunk_files = sorted(glob.glob(chunk_pattern))

#     if not chunk_files:
#         raise FileNotFoundError(f"No chunk files (*_chunk_*.npz) found in '{directory_path}'.")

#     print(f"Found {len(chunk_files)} chunk files to merge from '{directory_path}'.")

#     # --- Step 1: First pass to count total trajectories ---
#     total_trajectories = 0
#     for file_path in tqdm(chunk_files, desc="Scanning chunks"):
#         try:
#             with np.load(file_path, allow_pickle=True) as data:
#                 total_trajectories += len(data["states"])
#         except Exception as e:
#             print(f"Warning: Could not read {file_path}, skipping. Error: {e}")

#     if total_trajectories == 0:
#         raise ValueError("No trajectories could be loaded from the chunk files.")

#     print(f"Found a total of {total_trajectories} trajectories. Pre-allocating memory...")

#     # --- Step 2: Pre-allocate numpy object arrays ---
#     all_states = np.empty(total_trajectories, dtype=object)
#     all_actions = np.empty(total_trajectories, dtype=object)
#     all_next_states = np.empty(total_trajectories, dtype=object)
#     all_dones = np.empty(total_trajectories, dtype=object)

#     # --- Step 3: Second pass to fill the arrays ---
#     current_idx = 0
#     for file_path in tqdm(chunk_files, desc="Merging chunks"):
#         try:
#             with np.load(file_path, allow_pickle=True) as data:
                
#                 # FIX: Handle both object arrays and dense N-D arrays
#                 def fill_array(target_array, source_data):
#                     chunk_size = len(source_data)
#                     # If numpy "helpfully" created a dense array, iterate through it
#                     if source_data.ndim > 1 and source_data.dtype != object:
#                         for i in range(chunk_size):
#                             target_array[current_idx + i] = source_data[i]
#                     # Otherwise, it's an object array, assign the slice directly
#                     else:
#                         target_array[current_idx : current_idx + chunk_size] = source_data
#                     return chunk_size

#                 chunk_size = fill_array(all_states, data["states"])
#                 fill_array(all_actions, data["actions"])
#                 fill_array(all_next_states, data["next_states"])
#                 fill_array(all_dones, data["dones"])
                
#                 current_idx += chunk_size
#         except Exception as e:
#             print(f"Warning: Error processing {file_path}, skipping. Error: {e}")
#             continue
    
#     print(f"Successfully merged {current_idx} trajectories in memory.")

#     return {
#         "states": all_states,
#         "actions": all_actions,
#         "next_states": all_next_states,
#         "dones": all_dones
#     }

# @hydra.main(version_base=None, config_path=".", config_name="train_f450")
# def main(cfg):
#     OmegaConf.register_new_resolver("eval", eval)
#     OmegaConf.resolve(cfg)
#     OmegaConf.set_struct(cfg, False)
#     simulation_app = init_simulation_app(cfg)
#     run = init_wandb(cfg)
#     setproctitle(run.name)
#     print(OmegaConf.to_yaml(cfg))

#     from omni_drones.envs.isaac_env import IsaacEnv

#     env_class = IsaacEnv.REGISTRY[cfg.task.name]
#     base_env = env_class(cfg, headless=cfg.headless)

#     num_framestack = cfg.get("num_framestack", -1)
#     if num_framestack > 0:
#         # ========================================================================
#         # ADD THE FRAMESTACK TRANSFORM HERE
#         # ========================================================================
#         transforms = [
#             InitTracker(),
#             # This transform will take the last 4 observations from ("agents", "observation")
#             # and stack them along a new dimension, creating a new key.
#             # Let's call the new output key "observation_stacked".
#             CatFrames(
#                 N=num_framestack, 
#                 in_keys=[("agents", "observation")], 
#                 out_keys=[("agents", "observation_stacked")]
#             )
#         ]
#         # ========================================================================

#     # ==================== MODIFICATION: Cyclical Chunk Loading Setup ====================
#     # --- CONFIGURATION ---
#     # Train on each chunk for this many iterations before loading the next one.
#     TRAINING_ITERATIONS_PER_CHUNK = 25 #100 #50 #25
#     # ---------------------

#     data_dir = "deltaAction_trainingDataset"
#     base_path = os.path.dirname(__file__)
#     full_data_path = os.path.join(base_path, data_dir)

#     if not os.path.isdir(full_data_path):
#         raise FileNotFoundError(f"Data directory not found: {full_data_path}")
    
#     # Get a sorted list of all chunk files to ensure a consistent order
#     chunk_files = sorted([os.path.join(full_data_path, f) for f in os.listdir(full_data_path) if f.endswith('.npz')])
    
#     if not chunk_files:
#         raise FileNotFoundError(f"No '.npz' data files found in '{full_data_path}'.")

#     print(f"Found {len(chunk_files)} data chunks to cycle through.")
    
#     # Initialize a counter for which chunk to load next
#     chunk_load_counter = 0

#     # --- Perform the initial data load using the first chunk ---
#     base_env.load_data_from_chunk(chunk_files[0])
#     # ====================================================================================

#     transforms = [InitTracker()]

#     # a CompositeSpec is by default processed by a entity-based encoder
#     if num_framestack > 0:
#         # a CompositeSpec is by default processed by a entity-based encoder
#         # ravel it to use a MLP encoder instead
#         if cfg.task.get("ravel_obs", False):
#             # NOTE: The ravel transform now needs to act on our new stacked observation!
#             # It's better to handle the flattening inside the policy network itself.
#             # Let's comment this out for now and handle it in PPO.py.
#             # transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
#             # transforms.append(transform)
#             pass # We will handle flattening in the policy
#     else:
#         # ravel it to use a MLP encoder instead
#         if cfg.task.get("ravel_obs", False):
#             transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
#             transforms.append(transform)
#         if cfg.task.get("ravel_obs_central", False):
#             transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
#             transforms.append(transform)

#     # optionally discretize the action space or use a controller
#     action_transform: str = cfg.task.get("action_transform", None)
#     if action_transform is not None:
#         if action_transform.startswith("multidiscrete"):
#             nbins = int(action_transform.split(":")[1])
#             transform = FromMultiDiscreteAction(nbins=nbins)
#             transforms.append(transform)
#         elif action_transform.startswith("discrete"):
#             nbins = int(action_transform.split(":")[1])
#             transform = FromDiscreteAction(nbins=nbins)
#             transforms.append(transform)
#         else:
#             raise NotImplementedError(f"Unknown action transform: {action_transform}")

#     env = TransformedEnv(base_env, Compose(*transforms)).train()
#     env.set_seed(cfg.seed)

#     try:
#         policy = ALGOS[cfg.algo.name.lower()](
#             cfg.algo,
#             env.observation_spec,
#             env.action_spec,
#             env.reward_spec,
#             device=base_env.device
#         )
#     except KeyError:
#         raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

#     # '''
#     # Path of the directory containing train_f450.py
#     base_dir = os.path.dirname(os.path.abspath(__file__))

#     # Build absolute path to the pretrained checkpoint
#     pretrained_ckpt_name = cfg.get("pretrained_ckpt_path", None)
#     if pretrained_ckpt_name:
#         pretrained_ckpt_path = os.path.join(base_dir, pretrained_ckpt_name)
#         if os.path.exists(pretrained_ckpt_path):
#             print(f"Loading pretrained model from {pretrained_ckpt_path}")
#             policy.load_state_dict(torch.load(pretrained_ckpt_path, map_location=base_env.device, weights_only=True))
#         else:
#             print(f"[Warning] Checkpoint path {pretrained_ckpt_path} not found.")
#     # '''

#     frames_per_batch = env.num_envs * int(cfg.algo.train_every)
#     total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
#     max_iters = cfg.get("max_iters", -1)
#     eval_interval = cfg.get("eval_interval", -1)
#     save_interval = cfg.get("save_interval", -1)

#     stats_keys = [
#         k for k in base_env.observation_spec.keys(True, True)
#         if isinstance(k, tuple) and k[0]=="stats"
#     ]
#     episode_stats = EpisodeStats(stats_keys)
#     collector = SyncDataCollector(
#         env,
#         policy=policy,
#         frames_per_batch=frames_per_batch,
#         total_frames=total_frames,
#         device=cfg.sim.device,
#         return_same_td=True,
#     )
#     best_return_value, return_value = base_env.max_episode_length*0., 0.  # to save the best model
    
#     # print(env.input_spec)
#     # print(env.output_spec)

#     @torch.no_grad()
#     def evaluate(
#         seed: int=0,
#         exploration_type: ExplorationType=ExplorationType.MODE
#     ):

#         base_env.enable_render(True)
#         base_env.eval()
#         env.eval()
#         env.set_seed(seed)

#         render_callback = RenderCallback(interval=2)

#         with set_exploration_type(exploration_type):
#             trajs = env.rollout(
#                 max_steps=base_env.max_episode_length,
#                 policy=policy,
#                 callback=render_callback,
#                 auto_reset=True,
#                 break_when_any_done=False,
#                 return_contiguous=False,
#             )
#         base_env.enable_render(not cfg.headless)
#         env.reset()

#         done = trajs.get(("next", "done"))
#         first_done = torch.argmax(done.long(), dim=1).cpu()

#         def take_first_episode(tensor: torch.Tensor):
#             indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
#             return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

#         traj_stats = {
#             k: take_first_episode(v)
#             for k, v in trajs[("next", "stats")].cpu().items()
#         }

#         info = {
#             "eval/stats." + k: torch.mean(v.float()).item()
#             for k, v in traj_stats.items()
#         }

#         # log video
#         info["recording"] = wandb.Video(
#             render_callback.get_video_array(axes="t c h w"),
#             fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
#             format="mp4"
#         )

#         # log distributions
#         # df = pd.DataFrame(traj_stats)
#         # table = wandb.Table(dataframe=df)
#         # info["eval/return"] = wandb.plot.histogram(table, "return")
#         # info["eval/episode_len"] = wandb.plot.histogram(table, "episode_len")

#         return info

#     pbar = tqdm(collector, total=total_frames//frames_per_batch)
#     env.train()
#     for i, data in enumerate(pbar):
#         info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
#         episode_stats.add(data.to_tensordict())

#         if len(episode_stats) >= base_env.num_envs:
#             stats = {
#                 "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
#                 for k, v in episode_stats.pop().items(True, True)
#             }
            
#             return_value = stats.get('train/stats.return', 0)  #return_value = stats['train/stats.return']
#             ey = stats.get('train/stats.ey', 0) 
#             '''
#             rwd_ey = stats.get('train/stats.rwd_ey', 0) 
#             rwd_ey_dot = stats.get('train/stats.rwd_ey_dot', 0) 
#             rwd_eq = stats.get('train/stats.rwd_eq', 0) 
#             # rwd_ew = stats.get('train/stats.rwd_ew', 0) 
#             # rwd_eR = stats.get('train/stats.rwd_eR', 0) 
#             rwd_eb1 = stats.get('train/stats.rwd_eb1', 0) 
#             # rwd_eW = stats.get('train/stats.rwd_eW', 0) 

#             # Define the balanced score. 
#             balanced_score = 1.5 * rwd_ey + \
#                              0.7 * rwd_ey_dot + \
#                              0.5 * rwd_eq + \
#                              0.8 * rwd_eb1
#                             # + rwd_ew + rwd_eR #+ rwd_eW
#             info["train/stats.balanced_score"] = balanced_score
#             '''

#             info.update(stats)

#             # if save_interval > 0 and i % save_interval == 0:
#             if save_interval > 0 and ey < 0.1 and return_value > best_return_value:
#                 try:
#                     ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}_best.pth")
#                     torch.save(policy.state_dict(), ckpt_path)
#                     logging.info(f"Saved checkpoint to {str(ckpt_path)}")
#                     print("return_value:", return_value, "best_return_value:", best_return_value)
#                     best_return_value = return_value
#                 except AttributeError:
#                     logging.warning(f"Policy {policy} does not implement `.state_dict()`")
        
#         if i % 250 == 0:
#             ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pth")
#             torch.save(policy.state_dict(), ckpt_path)
#             logging.info(f"Saved delta action model checkpoint to {ckpt_path}")

#         info.update(policy.train_op(data.to_tensordict(),i))

#         # Log current learning rates ---
#         info["train/lr_a"] = policy.actor_opt.param_groups[0]['lr']
#         info["train/lr_c"] = policy.critic_opt.param_groups[0]['lr']
#         """
#         # --- NEW LOGIC: Periodically load the next data chunk ---
#         if i > 0 and i % TRAINING_ITERATIONS_PER_CHUNK == 0:
#             logging.info(f"Iteration {i}: Swapping data chunk.")

#             # Step 1: THE FIX - Reset the collector FIRST.
#             # This empties the collector's internal buffer, releasing its
#             # reference to the old data and preventing a memory leak.
#             collector.reset()

#             # Step 2: Determine the next chunk to load.
#             chunk_load_counter += 1
#             chunk_index_to_load = chunk_load_counter % len(chunk_files)
            
#             # Step 3: Now, load the new chunk.
#             # The 'load_data_from_chunk' function should contain the 'gc.collect()'
#             # call, which will now be effective because the collector's reference is gone.
#             base_env.load_data_from_chunk(chunk_files[chunk_index_to_load])
#         # -----------------------------------------------------------
#         """
#         # --- NEW LOGIC: Periodically load the next data chunk ---
#         if i > 0 and i % TRAINING_ITERATIONS_PER_CHUNK == 0:
#             logging.info(f"Iteration {i}: Swapping data chunk.")
            
#             # Step 1: Determine the next chunk to load.
#             chunk_load_counter += 1
#             chunk_index_to_load = chunk_load_counter % len(chunk_files)
            
#             # Step 2: Load the new chunk into the environment.
#             base_env.load_data_from_chunk(chunk_files[chunk_index_to_load])

#             # Step 3: THE DEFINITIVE FIX - Reset the collector.
#             # This forces all environments to start new episodes from t=0,
#             # which resets all internal timestep counters and prevents the
#             # "index out of bounds" error.
#             collector.reset()
#         # -----------------------------------------------------------

#         if eval_interval > 0 and i % eval_interval == 0:
#             logging.info(f"Eval at {collector._frames} steps.")
#             info.update(evaluate())
#             env.train()
#             base_env.train()

#         run.log(info)
#         print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

#         pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

#         if max_iters > 0 and i >= max_iters - 1:
#             break

#     logging.info(f"Final Eval at {collector._frames} steps.")
#     info = {"env_frames": collector._frames}
#     info.update(evaluate())
#     run.log(info)

#     try:
#         ckpt_path = os.path.join(run.dir, "checkpoint_final.pth")
#         torch.save(policy.state_dict(), ckpt_path)

#         model_artifact = wandb.Artifact(
#             f"{cfg.task.name}-{cfg.algo.name.lower()}",
#             type="model",
#             description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
#             metadata=dict(cfg))

#         model_artifact.add_file(ckpt_path)
#         wandb.save(ckpt_path)
#         run.log_artifact(model_artifact)

#         logging.info(f"Saved checkpoint to {str(ckpt_path)}")
#     except AttributeError:
#         logging.warning(f"Policy {policy} does not implement `.state_dict()`")

#     wandb.finish()

#     simulation_app.close()


# if __name__ == "__main__":
#     main()

import logging
import os
import gc
import glob
import time

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

# --- THIS IS THE FIX (Part 1): Make sure random is imported ---
import random 
# -----------------------------------------------------------

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from torchrl.envs.transforms import CatFrames 

def load_and_merge_chunks(directory_path: str):
    """
    Finds all .npz chunks in a directory, merges them into a single 
    dataset in memory, and returns the data as a dictionary. This version
    is optimized for lower memory usage and handles different numpy array formats.
    """
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Provided data path is not a directory: {directory_path}")

    chunk_pattern = os.path.join(directory_path, "*_chunk_*.npz")
    chunk_files = sorted(glob.glob(chunk_pattern))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files (*_chunk_*.npz) found in '{directory_path}'.")

    print(f"Found {len(chunk_files)} chunk files to merge from '{directory_path}'.")

    # --- Step 1: First pass to count total trajectories ---
    total_trajectories = 0
    for file_path in tqdm(chunk_files, desc="Scanning chunks"):
        try:
            with np.load(file_path, allow_pickle=True) as data:
                total_trajectories += len(data["states"])
        except Exception as e:
            print(f"Warning: Could not read {file_path}, skipping. Error: {e}")

    if total_trajectories == 0:
        raise ValueError("No trajectories could be loaded from the chunk files.")

    print(f"Found a total of {total_trajectories} trajectories. Pre-allocating memory...")

    # --- Step 2: Pre-allocate numpy object arrays ---
    all_states = np.empty(total_trajectories, dtype=object)
    all_actions = np.empty(total_trajectories, dtype=object)
    all_next_states = np.empty(total_trajectories, dtype=object)
    all_dones = np.empty(total_trajectories, dtype=object)

    # --- Step 3: Second pass to fill the arrays ---
    current_idx = 0
    for file_path in tqdm(chunk_files, desc="Merging chunks"):
        try:
            with np.load(file_path, allow_pickle=True) as data:
                
                # FIX: Handle both object arrays and dense N-D arrays
                def fill_array(target_array, source_data):
                    chunk_size = len(source_data)
                    # If numpy "helpfully" created a dense array, iterate through it
                    if source_data.ndim > 1 and source_data.dtype != object:
                        for i in range(chunk_size):
                            target_array[current_idx + i] = source_data[i]
                    # Otherwise, it's an object array, assign the slice directly
                    else:
                        target_array[current_idx : current_idx + chunk_size] = source_data
                    return chunk_size

                chunk_size = fill_array(all_states, data["states"])
                fill_array(all_actions, data["actions"])
                fill_array(all_next_states, data["next_states"])
                fill_array(all_dones, data["dones"])
                
                current_idx += chunk_size
        except Exception as e:
            print(f"Warning: Error processing {file_path}, skipping. Error: {e}")
            continue
    
    print(f"Successfully merged {current_idx} trajectories in memory.")

    return {
        "states": all_states,
        "actions": all_actions,
        "next_states": all_next_states,
        "dones": all_dones
    }

@hydra.main(version_base=None, config_path=".", config_name="train_f450")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    num_framestack = cfg.get("num_framestack", -1)
    if num_framestack > 0:
        # ========================================================================
        # ADD THE FRAMESTACK TRANSFORM HERE
        # ========================================================================
        transforms = [
            InitTracker(),
            # This transform will take the last 4 observations from ("agents", "observation")
            # and stack them along a new dimension, creating a new key.
            # Let's call the new output key "observation_stacked".
            CatFrames(
                N=num_framestack, 
                in_keys=[("agents", "observation")], 
                out_keys=[("agents", "observation_stacked")]
            )
        ]
        # ========================================================================

    # ==================== MODIFICATION: Cyclical Chunk Loading Setup ====================
    # --- CONFIGURATION ---
    # Train on each chunk for this many iterations before loading the next one.
    TRAINING_ITERATIONS_PER_CHUNK = 30 #25 #20 #40 #15 #100 #50
    # ---------------------

    data_dir = "deltaAction_trainingDataset"
    base_path = os.path.dirname(__file__)
    full_data_path = os.path.join(base_path, data_dir)

    if not os.path.isdir(full_data_path):
        raise FileNotFoundError(f"Data directory not found: {full_data_path}")
    
    # --- THIS IS THE FIX (Part 2) ---
    # Get a list of all chunk files
    chunk_files = [os.path.join(full_data_path, f) for f in os.listdir(full_data_path) if f.endswith('.npz')]
    
    if not chunk_files:
        raise FileNotFoundError(f"No '.npz' data files found in '{full_data_path}'.")

    # Randomly shuffle the list of chunks
    random.shuffle(chunk_files)
    print(f"Found and SHUFFLED {len(chunk_files)} data chunks to cycle through.")
    # --- END FIX ---
    
    # Initialize a counter for which chunk to load next
    chunk_load_counter = 0

    # --- Perform the initial data load using the first *random* chunk ---
    base_env.load_data_from_chunk(chunk_files[0])
    # ====================================================================================

    transforms = [InitTracker()]

    # a CompositeSpec is by default processed by a entity-based encoder
    if num_framestack > 0:
        # a CompositeSpec is by default processed by a entity-based encoder
        # ravel it to use a MLP encoder instead
        if cfg.task.get("ravel_obs", False):
            # NOTE: The ravel transform now needs to act on our new stacked observation!
            # It's better to handle the flattening inside the policy network itself.
            # Let's comment this out for now and handle it in PPO.py.
            # transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
            # transforms.append(transform)
            pass # We will handle flattening in the policy
    else:
        # ravel it to use a MLP encoder instead
        if cfg.task.get("ravel_obs", False):
            transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
            transforms.append(transform)
        if cfg.task.get("ravel_obs_central", False):
            transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
            transforms.append(transform)

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
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
            device=base_env.device
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    '''
    # Path of the directory containing train_f450.py
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Build absolute path to the pretrained checkpoint
    pretrained_ckpt_name = cfg.get("pretrained_ckpt_path", None)
    if pretrained_ckpt_name:
        pretrained_ckpt_path = os.path.join(base_dir, pretrained_ckpt_name)
        if os.path.exists(pretrained_ckpt_path):
            print(f"Loading pretrained model from {pretrained_ckpt_path}")
            policy.load_state_dict(torch.load(pretrained_ckpt_path, map_location=base_env.device, weights_only=True))
        else:
            print(f"[Warning] Checkpoint path {pretrained_ckpt_path} not found.")
    '''

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )
    best_return_value, best_error_metric, return_value = base_env.max_episode_length*0., float('inf'), 0.  # to save the best model
    
    # print(env.input_spec)
    # print(env.output_spec)

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        exploration_type: ExplorationType=ExplorationType.MODE
    ):

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = RenderCallback(interval=2)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        # log video
        info["recording"] = wandb.Video(
            render_callback.get_video_array(axes="t c h w"),
            fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
            format="mp4"
        )

        # log distributions
        # df = pd.DataFrame(traj_stats)
        # table = wandb.Table(dataframe=df)
        # info["eval/return"] = wandb.plot.histogram(table, "return")
        # info["eval/episode_len"] = wandb.plot.histogram(table, "episode_len")

        return info

    pbar = tqdm(collector, total=total_frames//frames_per_batch)
    env.train()
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        saved_this_iter = False # Track saves to avoid redundancy

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            
            return_value = stats.get('train/stats.return', 0)  #return_value = stats['train/stats.return']
            ey = stats.get('train/stats.ey', 0) 
            eb1 = stats.get('train/stats.eb1', 0)
            '''
            rwd_ey = stats.get('train/stats.rwd_ey', 0) 
            rwd_ey_dot = stats.get('train/stats.rwd_ey_dot', 0) 
            rwd_eq = stats.get('train/stats.rwd_eq', 0) 
            # rwd_ew = stats.get('train/stats.rwd_ew', 0) 
            # rwd_eR = stats.get('train/stats.rwd_eR', 0) 
            rwd_eb1 = stats.get('train/stats.rwd_eb1', 0) 
            # rwd_eW = stats.get('train/stats.rwd_eW', 0) 

            # Define the balanced score. 
            balanced_score = 1.5 * rwd_ey + \
                             0.7 * rwd_ey_dot + \
                             0.5 * rwd_eq + \
                             0.8 * rwd_eb1
                             # + rwd_ew + rwd_eR #+ rwd_eW
            info["train/stats.balanced_score"] = balanced_score
            '''

            info.update(stats)

            # if save_interval > 0 and i % save_interval == 0:
            if save_interval > 0 and ey < 0.1 and return_value > best_return_value:
                best_return_value = return_value
                try:
                    ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}_best.pth")
                    torch.save(policy.state_dict(), ckpt_path)
                    logging.info(f"Saved checkpoint to {str(ckpt_path)}")
                    print("return_value:", return_value, "best_return_value:", best_return_value)
                    saved_this_iter = True
                except AttributeError:
                    logging.warning(f"Policy {policy} does not implement `.state_dict()`")
        
            # 2. Best error (ey + eb1)
            current_error_metric = ey + eb1
            if save_interval > 0 and current_error_metric < best_error_metric:
                best_error_metric = current_error_metric
                try:
                    ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}_best_error.pth")
                    torch.save(policy.state_dict(), ckpt_path)
                    logging.info(f"Saved best error checkpoint to {ckpt_path} (ey+eb1: {current_error_metric:.4f})")
                    saved_this_iter = True
                except AttributeError:
                    logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        if i % 250 == 0 and saved_this_iter is not True:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pth")
            torch.save(policy.state_dict(), ckpt_path)
            logging.info(f"Saved delta action model checkpoint to {ckpt_path}")

        info.update(policy.train_op(data.to_tensordict(),i))

        # Log current learning rates ---
        info["train/lr_a"] = policy.actor_opt.param_groups[0]['lr']
        info["train/lr_c"] = policy.critic_opt.param_groups[0]['lr']
        
        # --- NEW LOGIC: Periodically load the next data chunk ---
        if i > 0 and i % TRAINING_ITERATIONS_PER_CHUNK == 0:
            logging.info(f"Iteration {i}: Swapping data chunk.")
            
            # Step 1: Determine the next chunk to load.
            chunk_load_counter += 1
            chunk_index_to_load = chunk_load_counter % len(chunk_files)
            
            # Step 2: Load the new chunk into the environment.
            base_env.load_data_from_chunk(chunk_files[chunk_index_to_load])

            # Step 3: THE DEFINITIVE FIX - Reset the collector.
            # This forces all environments to start new episodes from t=0,
            # which resets all internal timestep counters and prevents the
            # "index out of bounds" error.
            collector.reset()
        # -----------------------------------------------------------

        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate())
            env.train()
            base_env.train()

        run.log(info)
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if max_iters > 0 and i >= max_iters - 1:
            break

    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate())
    run.log(info)

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pth")
        torch.save(policy.state_dict(), ckpt_path)

        model_artifact = wandb.Artifact(
            f"{cfg.task.name}-{cfg.algo.name.lower()}",
            type="model",
            description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
            metadata=dict(cfg))

        model_artifact.add_file(ckpt_path)
        wandb.save(ckpt_path)
        run.log_artifact(model_artifact)

        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()