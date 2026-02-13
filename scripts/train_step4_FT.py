import logging
import os
import time

import hydra
import torch
import numpy as np
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from omni_drones.utils.wandb import init_wandb
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

# In train_main_step4_fine-tuning.py

# A wrapper policy that combines the policy-to-tune and the frozen 6D delta model
class FineTuningPolicy(TensorDictModuleBase):
    def __init__(self, policy_to_tune, delta_policy, delta_action_noise_scale=0.0, delta_bias=None):
        super().__init__()
        self.policy_to_tune = policy_to_tune # The 4D policy we are training
        self.delta_policy = delta_policy   # The frozen 6D delta model
        self.delta_action_noise_scale = delta_action_noise_scale
        # Default to zeros if None
        if delta_bias is None:
            self.delta_bias = None
        else:
            # Ensure it's a tensor on the correct device (will be moved in forward if needed)
            # We register it as a buffer so it moves with the module
            self.register_buffer("delta_bias", torch.tensor(delta_bias, dtype=torch.float32))

        # Freeze the delta policy
        for param in self.delta_policy.parameters():
            param.requires_grad = False
            
        # Get dimensions for slicing
        self.base_action_dim = 4
        self.delta_action_dim = 6  # <-- CHANGE FROM 7 to 6
        self.force_action_dim = 2 # <-- ADD THIS (6 - 4)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # Ensure correct modes: policy_to_tune is training, delta_policy is eval
        self.policy_to_tune.train() 
        self.delta_policy.eval()

        # --- Step 1: Get the base 4D action and log_prob from policy_to_tune ---
        # Run the policy_to_tune. It adds 'action' (4D) and 'sample_log_prob' to the tensordict.
        self.policy_to_tune(tensordict) 
        
        # Store the original 4D action and its log_prob under new keys for PPO update later
        base_action_4d = tensordict.get(("agents", "action"))
        tensordict.set(("agents", "base_action_4d"), base_action_4d)
        if "sample_log_prob" in tensordict.keys(include_nested=True):
             tensordict.set(("agents", "base_log_prob"), tensordict.get("sample_log_prob"))

        # --- Step 2: Prepare observation for the 6D delta policy ---
        sim_obs = tensordict[("agents", "observation")]
        
        # Ensure base_action_4d has the same middle dimension (time or sequence) as sim_obs if needed
        # This might be necessary if sim_obs has shape [B, T, D] and base_action has [B, D]
        if base_action_4d.ndim < sim_obs.ndim:
             base_action_4d_expanded = base_action_4d.unsqueeze(1).expand_as(sim_obs[..., :self.base_action_dim])
        else:
             base_action_4d_expanded = base_action_4d

        delta_obs = torch.cat([sim_obs, base_action_4d_expanded], dim=-1)
        
        # Create a temporary TensorDict for the delta policy input
        delta_input_td = TensorDict(
            {"agents": {"observation": delta_obs}}, 
            batch_size=tensordict.batch_size, 
            device=tensordict.device
        )
        # Include intrinsics if the delta policy uses them (should match policy_to_tune)
        if ("agents", "intrinsics") in tensordict.keys(True, True):
             delta_input_td[("agents", "intrinsics")] = tensordict[("agents", "intrinsics")]

        # --- Step 3: Get the 6D delta action from the frozen delta policy ---
        with torch.no_grad():
            # Get the "non-precise" delta action
            delta_action_6d = self.delta_policy(delta_input_td)[("agents", "action")]
            
            # Add noise
            noise = torch.randn_like(delta_action_6d) * self.delta_action_noise_scale
            delta_action_6d = delta_action_6d + noise

            # Add Constant Bias
            if self.delta_bias is not None:
                # Broadcast bias [6] to match batch shape [Batch, 1, 6]
                # Ensure bias is on correct device
                if self.delta_bias.device != delta_action_6d.device:
                    self.delta_bias = self.delta_bias.to(delta_action_6d.device)
                
                bias_expanded = self.delta_bias.view(1, 1, -1).expand_as(delta_action_6d)
                delta_action_6d = delta_action_6d + bias_expanded

        # --- Step 4: Combine actions for the ENVIRONMENT STEP ---
        # The environment's _pre_sim_step expects the full 6D vector 
        # (corrected_controls + fictitious_forces) under the "action" key.
        
        # Split the delta action
        delta_controls_4d = delta_action_6d[..., :self.base_action_dim]
        fictitious_forces_2d = delta_action_6d[..., self.base_action_dim:]

        # Calculate the final 4D control action applied to motors
        corrected_controls_4d = torch.clamp(base_action_4d + delta_controls_4d, min=-1., max=1.)
        
        # Recombine into the 6D vector expected by the modified _pre_sim_step
        action_for_env_6d = torch.cat([corrected_controls_4d, fictitious_forces_2d], dim=-1)

        # Overwrite the original "action" key with this 6D vector for the environment step
        tensordict.set(("agents", "action"), action_for_env_6d)
        
        return tensordict
    
@hydra.main(version_base=None, config_path=".", config_name="train_f450")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    # Use the nominal environment from Step 1
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).train()
    env.set_seed(cfg.seed)

    # --- 1. CONFIGURE AND LOAD PRE-TRAINED POLICY (to be fine-tuned) ---
    print("--- Configuring Policy to Fine-Tune (4D) ---")
    # 1a. Start with the base config loaded by Hydra (cfg.algo)
    #     Make a deep copy to avoid modifying the original
    import copy
    cfg_policy_to_tune = copy.deepcopy(cfg.algo)

    # 1b. Explicitly set the parameters for the PRE-TRAINED policy
    #     This assumes your PPOConfig defaults are set for Step 3 (domain_adaptation=True)
    #     Adjust these values to match your ACTUAL pre-trained policy's settings.
    cfg_policy_to_tune.domain_adaptation = False # Pre-trained was not domain adaptation
    cfg_policy_to_tune.actor_hidden_dim = 32      # Set the correct small actor size
    cfg_policy_to_tune.critic_hidden_dim = 256    # Set the correct small critic size
    # Add any other hyperparameters that differed during pre-training (e.g., LR) if necessary

    # 1a. Clone the current environment's spec
    policy_to_tune_obs_spec = env.observation_spec.clone()
    
    # 1b. Calculate the original observation dimension (without the added action)
    original_obs_dim = 32 # You know this from your environment code
    
    # 1c. Create the correct TensorSpec with 32 dimensions
    original_observation_tensor_spec = UnboundedContinuousTensorSpec(
        shape=(*env.observation_spec[("agents", "observation")].shape[:-1], original_obs_dim),
        device=base_env.device
    )
    
    # 1d. Overwrite the observation entry in the cloned spec
    policy_to_tune_obs_spec[("agents", "observation")] = original_observation_tensor_spec
    # ==========================================================

    # 1e. Get the correct 4D action spec (your existing code is correct)
    base_action_spec_4d = base_env.drone.action_spec.unsqueeze(0).to(base_env.device)
    policy_to_tune_action_spec = CompositeSpec({
        "agents": {"action": base_action_spec_4d}
    }).expand(env.batch_size) 

    # 1f. Load the policy using the tailored config, the CORRECTED 32D obs spec, and 4D action spec
    policy_to_tune = ALGOS[cfg_policy_to_tune.name.lower()](
        cfg_policy_to_tune, 
        policy_to_tune_obs_spec, # Pass the CORRECTED 32D obs spec
        policy_to_tune_action_spec[("agents", "action")], 
        env.reward_spec, 
        device=base_env.device
    )
    # '''
    if os.path.exists(cfg.finetune.policy_ckpt_path):
        print(f"Loading pre-trained policy from {cfg.finetune.policy_ckpt_path}")
        state_dict = torch.load(cfg.finetune.policy_ckpt_path, weights_only=True)
        policy_to_tune.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Policy checkpoint not found at {cfg.finetune.policy_ckpt_path}")
    # '''
    
    # --- 2. CONFIGURE AND LOAD DELTA ACTION MODEL (to be frozen) ---
    print("\n--- Configuring Delta Action Model (6D) ---")

    # === THE FIX: Manually create the correct 36D observation spec ===
    # 2a. Define the dimensions
    sim_obs_dim = 32 # Base simulation observation size
    base_action_dim = 4 # Dimension of the action from policy_to_tune
    delta_obs_dim = sim_obs_dim + base_action_dim # Total dimension is 36

    # 2b. Create the CompositeSpec structure
    delta_obs_spec = env.observation_spec.clone() # Start with env structure

    # 2c. Create the correct TensorSpec for the 36D observation
    delta_observation_tensor_spec = UnboundedContinuousTensorSpec(
        shape=(*env.observation_spec[("agents", "observation")].shape[:-1], delta_obs_dim),
        device=base_env.device
    )

    # 2d. Overwrite the observation entry in the cloned spec
    delta_obs_spec[("agents", "observation")] = delta_observation_tensor_spec
    # ========================================================================

    # 2e. Create the correct 6D action spec (This part is correct)
    delta_action_dim = 6 
    delta_action_spec = CompositeSpec({
        "agents": {"action": UnboundedContinuousTensorSpec((1, delta_action_dim))}
    }).expand(env.batch_size).to(base_env.device)

    # 2f. Create a tailored config object for the DELTA policy (This part is correct)
    cfg_delta_policy = copy.deepcopy(cfg.algo)
    cfg_delta_policy.domain_adaptation = True
    # cfg_delta_policy.actor_hidden_dim = [512, 256, 128]
    # cfg_delta_policy.critic_hidden_dim = [512, 256, 128]

    # 2g. Load the delta policy using its specific config and the CORRECT specs
    delta_policy = ALGOS[cfg_delta_policy.name.lower()](
        cfg_delta_policy,
        delta_obs_spec, # Pass the manually created 36D spec
        delta_action_spec[("agents", "action")],
        env.reward_spec,
        device=base_env.device
    )
    if os.path.exists(cfg.finetune.delta_ckpt_path):
        print(f"Loading delta action model from {cfg.finetune.delta_ckpt_path}")
        state_dict = torch.load(cfg.finetune.delta_ckpt_path, weights_only=True)
        delta_policy.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Delta model checkpoint not found at {cfg.finetune.delta_ckpt_path}")

    # --- CREATE THE WRAPPER POLICY (Unchanged) ---
    noise_scale = cfg.finetune.get("delta_action_noise_scale", 0.0)
    print(f"Applying delta action noise with scale: {noise_scale}")

    # --- Get Bias from Config ---
    # Expecting a list like [0.0, 0.0, 0.0, 0.1, 0.0, 0.0] for +Mz
    delta_bias_list = cfg.finetune.get("delta_bias", None)
    if delta_bias_list:
        print(f"Applying CONSTANT BIAS to delta action: {delta_bias_list}")
        # For +Mz, you want to add to index 3 (thrust=0, Mx=1, My=2, Mz=3, Fx=4, Fy=5)
        # Example in YAML: delta_bias: [0, 0, 0, 0.1, 0, 0]
    
    finetuning_policy = FineTuningPolicy(
        policy_to_tune, 
        delta_policy, 
        delta_action_noise_scale=noise_scale,
        delta_bias=delta_bias_list
    )

    # --- 4. SETUP DATA COLLECTION AND TRAINING ---
    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    save_interval = cfg.get("save_interval", -1)
    
    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    
    episode_stats = EpisodeStats(stats_keys)
    
    collector = SyncDataCollector(
        env,
        policy=finetuning_policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    best_return_value, best_error_metric, return_value = base_env.max_episode_length*0.8, float('inf'), 0.  # to save the best model

    # --- 4. SETUP TRAINING ---
    pbar = tqdm(collector, total=total_frames//frames_per_batch)
    env.train()
    for i, data in enumerate(pbar):
        # The 'data' object contains the buffer collected. The "action" key within it
        # is the original base_action from policy_to_tune, which is what we need for training.
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())
        
        saved_this_iter = False # Track saves to avoid redundancy

        if len(episode_stats) >= base_env.num_envs:
            # The stats are already nested, so we can iterate through them directly
            # stats = {
            #     "train/" + ".".join(k): v.mean().item()
            #     for k, v in episode_stats.pop().items(True, True)
            # }
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)
        
            # ADDED: Logic to save the best model based on return and position error
            return_value = stats.get('train/stats.return', -float('inf'))
            ey = stats.get('train/stats.ey', float('inf'))
            eb1 = stats.get('train/stats.eb1', float('inf'))

            if i > 3 and save_interval > 0 and return_value > best_return_value:# and ey < 0.1:
                best_return_value = return_value
                try:
                    ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}_best.pth")
                    torch.save(policy_to_tune.state_dict(), ckpt_path)
                    logging.info(f"Saved best fine-tuned checkpoint to {ckpt_path} (Return: {return_value:.2f})")
                    saved_this_iter = True
                except AttributeError:
                    logging.warning("Policy does not implement `.state_dict()`")

            # 2. Best error (ey + eb1)
            current_error_metric = ey + eb1
            if i > 3 and save_interval > 0 and current_error_metric < best_error_metric:
                best_error_metric = current_error_metric
                try:
                    ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}_best_error.pth")
                    torch.save(policy_to_tune.state_dict(), ckpt_path)
                    logging.info(f"Saved best error checkpoint to {ckpt_path} (ey+eb1: {current_error_metric:.4f})")
                    saved_this_iter = True
                except AttributeError:
                    logging.warning(f"Policy does not implement `.state_dict()`")

        if i > 3 and i % 3 == 0 and saved_this_iter is not True:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pth")
            # --- SAVE THE 4D ACTOR ---
            torch.save(policy_to_tune.state_dict(), ckpt_path)
            logging.info(f"Saved fine-tuned checkpoint to {ckpt_path}")

        # === THE CRUCIAL FIX FOR PPO UPDATE ===
        # Before calling train_op, restore the original 4D action and log_prob 
        # that the policy_to_tune actually produced during the rollout.
        data.set(
            ("agents", "action"), 
            data.get(("agents", "base_action_4d"))
        )
        if ("agents", "base_log_prob") in data.keys(True, True):
             data.set("sample_log_prob", data.get(("agents", "base_log_prob")))
        # ======================================

        # Train ONLY the policy_to_tune using its original actions and log_probs.
        train_info = policy_to_tune.train_op(data.to_tensordict(), i)
        info.update(train_info)
        
        # Log learning rates
        info["train/lr_a"] = policy_to_tune.actor_opt.param_groups[0]['lr']
        info["train/lr_c"] = policy_to_tune.critic_opt.param_groups[0]['lr']
        
        run.log(info)
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))
        
        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if max_iters > 0 and i >= max_iters - 1:
            break

    # --- 5. FINAL CHECKPOINT SAVE ---
    if run is not None:
        try:
            ckpt_path = os.path.join(run.dir, "checkpoint_final_finetuned.pth")
            torch.save(policy_to_tune.state_dict(), ckpt_path)
            logging.info(f"Saved final fine-tuned checkpoint to {ckpt_path}")
        except AttributeError:
            logging.warning("Policy does not implement `.state_dict()`")
    
    wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()