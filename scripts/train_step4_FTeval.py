import os
import copy
import hydra
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

# ==============================================================================
# ==== 1. PLOTTING & SUMMARY FUNCTIONS =========================================
# ==============================================================================

def plot_comparison_results(
    results_pretrained,
    results_finetuned,
    target_state,
    episode_num,
    save_dir
):
    """
    Plots a side-by-side comparison between the pre-trained and fine-tuned models.
    """
    states_pt, actions_pt, crashed_pt = results_pretrained["states"], results_pretrained["corrected_actions"], results_pretrained["crashed"]
    states_ft, actions_ft, crashed_ft = results_finetuned["states"], results_finetuned["corrected_actions"], results_finetuned["crashed"]
    time_pt, time_ft = np.arange(len(states_pt)), np.arange(len(states_ft))

    # --- Plot 1: Actions ---
    fig_act, axs_act = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig_act.suptitle(f'Action Comparison - Episode {episode_num}', fontsize=16)
    action_labels = ['Total Thrust (f)', 'Moment (M_1)', 'Moment (M_2)', 'Moment (M_3)']
    for i in range(4):
        axs_act[i].plot(time_pt, actions_pt[:, i], 'b-', alpha=0.6, label='Corrected Action (Pre-trained)')
        axs_act[i].plot(time_ft, actions_ft[:, i], 'r-', label='Corrected Action (Fine-tuned)')
        if crashed_pt: axs_act[i].scatter(time_pt[-1], actions_pt[-1, i], color='blue', s=35, zorder=5, label='Pre-trained Crash')
        if crashed_ft: axs_act[i].scatter(time_ft[-1], actions_ft[-1, i], color='red', s=35, zorder=5, label='Fine-tuned Crash')
        axs_act[i].set_ylabel(action_labels[i]); axs_act[i].legend(); axs_act[i].grid(True, linestyle=':')
    axs_act[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(save_dir, f"ep{episode_num}_comparison_acts.png"))
    plt.close(fig_act)

    # --- Plot 2, 3, 4: State Comparisons ---
    state_plots_info = {
        "Payload_State": (2, 3, [("Position, y", "y", "[m]"), ("Velocity, y_dot", "y_dot", "[m/s]")]),
        "Angular_States": (3, 3, [("Bar Direction, q", "q", ""), ("Bar Ang. Vel., w", "w", "[rad/s]"), ("Drone Ang. Vel., W", "W", "[rad/s]")]),
        "Rotation_Matrix": (3, 3, [("R_1x", "R", ""), ("R_2x", "R", ""), ("R_3x", "R", "")])
    }
    for title, (rows, cols, subplots) in state_plots_info.items():
        fig, axs = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)
        fig.suptitle(f'{title.replace("_", " ")} Comparison - Episode {episode_num}', fontsize=16)
        for i in range(rows):
            for j in range(cols):
                ax = axs[i, j] if rows > 1 else axs[j]
                
                if "R_" in subplots[i][0]:
                    matrix_row_idx = j
                    matrix_col_idx = i
                    base_idx = 12 + (matrix_row_idx * 3 + matrix_col_idx)
                    subtitle = f"R_{matrix_row_idx+1}{matrix_col_idx+1}"
                    target_val = target_state["R"][matrix_row_idx, matrix_col_idx]
                else:
                    subtitle_text, state_key, unit = subplots[i]
                    base_idx = {"y": 0, "y_dot": 3, "q": 6, "w": 9, "W": 21}[state_key] + j
                    subtitle = f'{subtitle_text}{["1", "2", "3"][j]} {unit}'
                    target_val = target_state[state_key][j]

                ax.axhline(y=target_val, color='k', linestyle='--', label='Target')
                ax.plot(time_pt, states_pt[:, base_idx], 'b-', alpha=0.6, label='Pre-trained')
                ax.plot(time_ft, states_ft[:, base_idx], 'r-', label='Fine-tuned')
                if crashed_pt: ax.scatter(time_pt[-1], states_pt[-1, base_idx], c='b', s=40, zorder=5)
                if crashed_ft: ax.scatter(time_ft[-1], states_ft[-1, base_idx], c='r', s=40, zorder=5)
                ax.set_title(subtitle); ax.legend(); ax.grid(True, linestyle=':')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(save_dir, f"ep{episode_num}_comparison_{title.lower()}.png"))
        plt.close(fig)

def plot_detailed_run(results, target_state, episode_num, save_dir, mode):
    """Plots the detailed trajectory and action composition for a single run."""
    states, base_actions, delta_actions, corrected_actions, crashed = (
        results["states"], results["base_actions"], results["delta_actions"], 
        results["corrected_actions"], results["crashed"]
    )
    if len(states) == 0: return # Skip plotting if no data
    time = np.arange(len(states))
    color = 'r' if mode == "Fine-tuned" else 'b'
    
    # --- Plot 1: Action Composition ---
    fig_act, axs_act = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    fig_act.suptitle(f'Action Composition ({mode}) - Episode {episode_num}', fontsize=16)
    action_labels = ['Thrust (f)', 'Moment (M_1)', 'Moment (M_2)', 'Moment (M_3)']
    for i in range(4):
        axs_act[i].plot(time, delta_actions[:, i], 'g--', alpha=0.5, label='Delta Action from froze Δa model')
        axs_act[i].plot(time, base_actions[:, i], 'b-', label='Base Action from fine-tuned model')
        axs_act[i].plot(time, corrected_actions[:, i], 'r--', alpha=0.5, label='Corrected Action')
        if crashed: axs_act[i].scatter(time[-1], corrected_actions[-1, i], color='red', s=40, zorder=5, label='Crash Point')
        axs_act[i].set_ylabel(action_labels[i]); axs_act[i].legend(); axs_act[i].grid(True, linestyle=':')
    axs_act[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(save_dir, f"ep{episode_num}_{mode}_actions.png"))
    plt.close(fig_act)
    
    # --- Plot 2, 3, 4: State Trajectories ---
    state_plots_info = {
        "Payload_State": (2, 3, [("Position, y", "y", "[m]"), ("Velocity, y_dot", "y_dot", "[m/s]")]),
        "Angular_States": (3, 3, [("Bar Direction, q", "q", ""), ("Bar Ang. Vel., w", "w", "[rad/s]"), ("Drone Ang. Vel., W", "W", "[rad/s]")]),
        "Rotation_Matrix": (3, 3, [("R_1x", "R", ""), ("R_2x", "R", ""), ("R_3x", "R", "")])
    }
    for title, (rows, cols, subplots_info) in state_plots_info.items():
        fig, axs = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)
        fig.suptitle(f'{title.replace("_", " ")} ({mode}) - Episode {episode_num}', fontsize=16)
        for i in range(rows):
            for j in range(cols):
                ax = axs[i, j] if rows > 1 else axs[j]
                
                if "R_" in subplots_info[i][0]:
                    matrix_col_idx, matrix_row_idx = j, i
                    base_idx = 12 + (matrix_row_idx * 3 + matrix_col_idx)
                    subtitle = f"R_{matrix_col_idx+1}{matrix_row_idx+1}"
                    target_val = target_state["R"][matrix_row_idx, matrix_col_idx]
                else:
                    sub_title, state_key, unit = subplots_info[i]
                    base_idx = {"y": 0, "y_dot": 3, "q": 6, "w": 9, "W": 21}[state_key] + j
                    subtitle = f'{sub_title}{["1", "2", "3"][j]} {unit}'
                    target_val = target_state[state_key][j]

                ax.axhline(y=target_val, color='k', linestyle='--', label='Target')
                ax.plot(time, states[:, base_idx], color=color, label=mode)
                if crashed: ax.scatter(time[-1], states[-1, base_idx], c='red', s=40, zorder=5, label='Crash Point')
                ax.set_title(subtitle); ax.legend(); ax.grid(True, linestyle=':')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(save_dir, f"ep{episode_num}_{mode}_{title.lower()}.png"))
        plt.close(fig)

def calculate_errors_at_checkpoints(sim_states, target_state, checkpoints_s, dt):
    """Calculates both average and steady-state errors at given time checkpoints."""
    results = {}
    if len(sim_states) == 0: return results

    for time_s in checkpoints_s:
        step_limit = int(time_s / dt)
        if step_limit > len(sim_states): continue

        sim_slice_avg = sim_states[:step_limit]
        ss_start_step = int(step_limit * 0.8)
        sim_slice_ss = sim_states[ss_start_step:step_limit]

        if len(sim_slice_ss) == 0: continue

        errors = {}
        for error_type, sim_slice in [("Avg", sim_slice_avg), ("SS", sim_slice_ss)]:
            target_y = np.tile(target_state["y"], (len(sim_slice), 1))
            target_ydot = np.tile(target_state["y_dot"], (len(sim_slice), 1))
            target_q = np.tile(target_state["q"], (len(sim_slice), 1))
            target_w = np.tile(target_state["w"], (len(sim_slice), 1))
            target_W = np.tile(target_state["W"], (len(sim_slice), 1))

            errors[f"{error_type} Pos Err (cm)"] = np.mean(np.linalg.norm(sim_slice[:, 0:3] - target_y, axis=1)) * 100
            errors[f"{error_type} Vel Err (cm/s)"] = np.mean(np.linalg.norm(sim_slice[:, 3:6] - target_ydot, axis=1)) * 100
            dot_q = np.einsum('ij,ij->i', sim_slice[:, 6:9], target_q)
            errors[f"{error_type} Bar Angle Err (deg)"] = np.mean(np.degrees(np.arccos(np.clip(dot_q, -1.0, 1.0))))
            errors[f"{error_type} Bar Vel Err (rad/s)"] = np.mean(np.linalg.norm(sim_slice[:, 9:12] - target_w, axis=1))
            errors[f"{error_type} Drone Vel Err (rad/s)"] = np.mean(np.linalg.norm(sim_slice[:, 21:24] - target_W, axis=1))
        
        results[time_s] = errors
    return results

def generate_summary_table(all_results, checkpoints_s, save_dir):
    """Generates a comprehensive summary table formatted like the example."""
    if not all_results: return

    report_parts = ["===== Evaluation Summary ====="]
    
    # --- Part 1: Per-Episode Breakdown ---
    for i, res_pair in enumerate(all_results):
        report_parts.append(f"\n--- Episode {i} ---")
        for mode, results in [("Pre-trained", res_pair["pretrained"]), ("Fine-tuned", res_pair["finetuned"])]:
            report_parts.append(f"\n  {mode}:")
            if results["crashed"]:
                report_parts.append(f"  Status: CRASHED at step {len(results['states'])}.")
            else:
                report_parts.append("  Status: SUCCESSFUL")
                if results["errors"]:
                    df_data = {"Length": [f"{s}s" for s in checkpoints_s]}
                    metric_names = list(results['errors'][checkpoints_s[0]].keys())
                    for metric in metric_names:
                        df_data[metric] = [results['errors'][s].get(metric, 'N/A') for s in checkpoints_s]
                    df = pd.DataFrame(df_data)
                    report_parts.append(df.to_string(index=False, float_format='%.2f'))

    # --- Part 2: Aggregated Summary ---
    report_parts.append("\n\n" + "="*30 + "\n          AGGREGATED RESULTS\n" + "="*30 + "\n")
    
    num_total = len(all_results)
    num_pt_success = sum(1 for res in all_results if not res["pretrained"]["crashed"])
    num_ft_success = sum(1 for res in all_results if not res["finetuned"]["crashed"])
    report_parts.append(f"Pre-trained Success Rate: {num_pt_success / num_total:.1%} ({num_pt_success}/{num_total})")
    report_parts.append(f"Fine-tuned Success Rate:  {num_ft_success / num_total:.1%} ({num_ft_success}/{num_total})\n")

    first_errors = next((r["pretrained"]["errors"] or r["finetuned"]["errors"] for r in all_results), None)
    metric_names = list(first_errors.get(checkpoints_s[0], {}).keys()) if first_errors else []

    if not metric_names:
        report_parts.append("No successful trajectories to calculate error metrics.")
    else:
        agg_table_data = {"Length": [f"{s}s" for s in checkpoints_s]}
        for metric in metric_names:
            agg_table_data[f"Pre-trained {metric}"] = []
            agg_table_data[f"Fine-tuned {metric}"] = []

        for time_s in checkpoints_s:
            for metric in metric_names:
                pt_vals = [r["pretrained"]["errors"][time_s][metric] for r in all_results if not r["pretrained"]["crashed"] and time_s in r["pretrained"]["errors"]]
                ft_vals = [r["finetuned"]["errors"][time_s][metric] for r in all_results if not r["finetuned"]["crashed"] and time_s in r["finetuned"]["errors"]]
                agg_table_data[f"Pre-trained {metric}"].append(f"{np.mean(pt_vals):.2f} ± {np.std(pt_vals):.2f}" if pt_vals else "N/A")
                agg_table_data[f"Fine-tuned {metric}"].append(f"{np.mean(ft_vals):.2f} ± {np.std(ft_vals):.2f}" if ft_vals else "N/A")
        
        df_agg = pd.DataFrame(agg_table_data)
        report_parts.append(df_agg.to_string(index=False))
        report_parts.append("\nError metrics are calculated over successful trajectories only.")

    final_report = "\n".join(report_parts)
    print("\n" + final_report)
    with open(os.path.join(save_dir, "evaluation_summary_report.txt"), "w") as f: f.write(final_report)

# ==============================================================================
# ==== 2. HELPER ROLLOUT FUNCTION ==============================================
# ==============================================================================

# In train_main_step4_evaluate_finetuning.py

def run_episode(policy_to_run, delta_policy, base_env, cfg_task):
    """
    Runs one full episode, with an option to smooth the base action input
    to the delta model for diagnostic purposes.
    """
    histories = {
        "states": [], "base_actions": [], "delta_actions": [], 
        "corrected_actions": [], "fictitious_forces": []
    }
    # --- THE FIX: SET BOTH POLICIES TO EVAL MODE ---
    policy_to_run.eval()
    delta_policy.eval()
    # ----------------------------------------------

    obs_td = base_env.reset()
    crashed = False

    # --- Smoothing Setup ---
    smooth_base_action_for_delta = False # Set to False to disable smoothing
    smoothing_alpha = 0.9             # Smoothing factor (0=no smooth, 1=use only previous)
    smoothed_base_action = None       # Initialize buffer
    # ----------------------

    # Get physical parameters for scaling plots
    # ... (your existing code to get limits)

    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        for t in range(base_env.max_episode_length):
            # --- Action Calculation ---
            # 1. Get 4D action from the base policy (potentially noisy)
            base_action_4d_raw = policy_to_run(obs_td)[("agents", "action")]

            # --- THE FIX: Smooth the base action *only* for delta input ---
            base_action_for_delta_input = base_action_4d_raw # Default
            if smooth_base_action_for_delta:
                if smoothed_base_action is None:
                    smoothed_base_action = base_action_4d_raw # Initialize on first step
                else:
                    smoothed_base_action = smoothing_alpha * smoothed_base_action + \
                                           (1 - smoothing_alpha) * base_action_4d_raw
                base_action_4d_raw = smoothed_base_action
            # --- END FIX ---

            # 2. Prepare observation for the 7D delta policy using the (potentially smoothed) base action
            sim_obs = obs_td[("agents", "observation")]
            
            # Use the SMOOTHED action here
            base_action_expanded = base_action_4d_raw if base_action_4d_raw.ndim == sim_obs.ndim else base_action_4d_raw.unsqueeze(1)
            delta_obs = torch.cat([sim_obs, base_action_expanded], dim=-1)
            delta_td = TensorDict({"agents": {"observation": delta_obs}}, batch_size=obs_td.batch_size, device=base_env.device)
            if ("agents", "intrinsics") in obs_td.keys(True, True):
                delta_td[("agents", "intrinsics")] = obs_td[("agents", "intrinsics")]
            
            # 3. Get 7D delta action (output might be smoother now)
            delta_action_6d = delta_policy(delta_td)[("agents", "action")]
            # delta_action_7d = delta_policy(delta_td)[("agents", "action")]

            # 4. Combine for the environment step using the ORIGINAL RAW base action
            delta_controls_4d = delta_action_6d[..., :4]
            fictitious_forces_2d = delta_action_6d[..., 4:]
            # delta_controls_4d = delta_action_7d[..., :4]
            # fictitious_forces_3d = delta_action_7d[..., 4:]

            # Use the RAW base action here for the final correction
            corrected_controls_4d = torch.clamp(base_action_4d_raw + delta_controls_4d, min=-1., max=1.)
            action_for_env_6d = torch.cat([corrected_controls_4d, fictitious_forces_2d], dim=-1)
            # action_for_env_7d = torch.cat([corrected_controls_4d, fictitious_forces_3d], dim=-1)
            
            # 5. Step the environment
            step_td = base_env.step(TensorDict({"agents": {"action": action_for_env_6d}}, batch_size=obs_td.batch_size))
            # step_td = base_env.step(TensorDict({"agents": {"action": action_for_env_7d}}, batch_size=obs_td.batch_size))

            # 6. Store histories (store RAW base action for accurate plotting)
            histories["states"].append(base_env.get_state())
            histories["base_actions"].append(base_action_4d_raw.squeeze().cpu().numpy()) # Store RAW
            histories["delta_actions"].append(delta_action_6d.squeeze().cpu().numpy())
            # histories["delta_actions"].append(delta_action_7d.squeeze().cpu().numpy())
            histories["corrected_actions"].append(corrected_controls_4d.squeeze().cpu().numpy())
            histories["fictitious_forces"].append(fictitious_forces_2d.squeeze().cpu().numpy())
            # histories["fictitious_forces"].append(fictitious_forces_3d.squeeze().cpu().numpy())

            if step_td[("next", "done")].any():
                crashed = t < (base_env.max_episode_length - 1)
                break
            
            obs_td = step_td["next"]
            
    for key, value in histories.items():
        # Ensure correct shape even if list is empty
        # expected_dim = {'states': 24, 'base_actions': 4, 'delta_actions': 7, 
        #                 'corrected_actions': 4, 'fictitious_forces': 3}.get(key, 1)
        expected_dim = {'states': 24, 'base_actions': 4, 'delta_actions': 6, 
                        'corrected_actions': 4, 'fictitious_forces': 2}.get(key, 1)
        histories[key] = np.array(value) if value else np.empty((0, expected_dim))
    
    histories["crashed"] = crashed
    return histories

# def run_episode(policy_to_run, delta_policy, base_env, cfg_task):
#     """
#     Runs one full episode using a base policy (pre-trained or fine-tuned)
#     combined with the frozen 7D delta model.
#     """
#     histories = {
#         "states": [], "base_actions": [], "delta_actions": [], 
#         "corrected_actions": [], "fictitious_forces": []
#     }
#     obs_td = base_env.reset()
#     crashed = False

#     # Get physical parameters for scaling plots
#     # (Add your get_thrust_limits and max_torque logic here if needed for plotting)

#     with set_exploration_type(ExplorationType.MODE), torch.no_grad():
#         for t in range(base_env.max_episode_length):
#             # --- Action Calculation ---
#             # 1. Get 4D action from the base policy
#             base_action_4d = policy_to_run(obs_td)[("agents", "action")]
            
#             # 2. Prepare observation for the 7D delta policy
#             sim_obs = obs_td[("agents", "observation")]
#             base_action_expanded = base_action_4d if base_action_4d.ndim == sim_obs.ndim else base_action_4d.unsqueeze(1)
#             delta_obs = torch.cat([sim_obs, base_action_expanded], dim=-1)
#             delta_td = TensorDict({"agents": {"observation": delta_obs}}, batch_size=obs_td.batch_size, device=base_env.device)
#             if ("agents", "intrinsics") in obs_td.keys(True, True):
#                 delta_td[("agents", "intrinsics")] = obs_td[("agents", "intrinsics")]
            
#             # 3. Get 7D delta action
#             delta_action_7d = delta_policy(delta_td)[("agents", "action")]

#             # 4. Combine for the environment step
#             delta_controls_4d = delta_action_7d[..., :4]
#             fictitious_forces_3d = delta_action_7d[..., 4:]
#             corrected_controls_4d = torch.clamp(base_action_4d + delta_controls_4d, min=-1., max=1.)
#             action_for_env_7d = torch.cat([corrected_controls_4d, fictitious_forces_3d], dim=-1)
            
#             # 5. Step the environment
#             step_td = base_env.step(TensorDict({"agents": {"action": action_for_env_7d}}, batch_size=obs_td.batch_size))

#             # 6. Store histories for plotting
#             histories["states"].append(base_env.get_state())
#             histories["base_actions"].append(base_action_4d.squeeze().cpu().numpy())
#             histories["delta_actions"].append(delta_action_7d.squeeze().cpu().numpy())
#             histories["corrected_actions"].append(corrected_controls_4d.squeeze().cpu().numpy())
#             histories["fictitious_forces"].append(fictitious_forces_3d.squeeze().cpu().numpy())

#             if step_td[("next", "done")].any():
#                 crashed = t < (base_env.max_episode_length - 1)
#                 break
            
#             obs_td = step_td["next"]
            
#     for key, value in histories.items():
#         histories[key] = np.array(value) if value else np.empty((0,1)) # Handle empty lists
    
#     histories["crashed"] = crashed
#     return histories

# ==============================================================================
# ==== 3. MAIN FUNCTION ========================================================
# ==============================================================================

@hydra.main(version_base=None, config_path=".", config_name="train_f450")
def main(cfg):
    
    # --- 1. Setup ---
    OmegaConf.resolve(cfg); OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs.isaac_env import IsaacEnv
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=True)
        
    # --- 2. Load All Three Policies ---
    
    # 2a. Load PRE-TRAINED Policy (Small or Large Arch, 4D Output)
    print("--- Loading Pre-trained Policy (4D) ---")
    cfg_policy_pt = copy.deepcopy(cfg.algo)
    # Ensure config matches the actual pre-trained model's settings
    cfg_policy_pt.domain_adaptation = False
    cfg_policy_pt.actor_hidden_dim = 32 #[512, 256, 128] # or 32 if it's the small one
    cfg_policy_pt.critic_hidden_dim = 256 #[512, 256, 128] # or 256 if it's the small one
    policy_pretrained_obs_spec = base_env.observation_spec.clone() # Base 32D obs
    base_action_spec_4d = base_env.drone.action_spec.unsqueeze(0).to(base_env.device)
    policy_pretrained_action_spec = CompositeSpec({"agents": {"action": base_action_spec_4d}}).expand(base_env.num_envs)
    
    policy_pretrained = ALGOS[cfg_policy_pt.name.lower()](
        cfg_policy_pt, policy_pretrained_obs_spec, policy_pretrained_action_spec[("agents", "action")], 
        base_env.reward_spec, device=base_env.device
    )
    if os.path.exists(cfg.eval.pretrained_ckpt_path):
        policy_pretrained.load_state_dict(torch.load(cfg.eval.pretrained_ckpt_path, weights_only=True))
    else: raise FileNotFoundError(f"Pre-trained policy not found at {cfg.eval.pretrained_ckpt_path}")
    policy_pretrained.eval()

    # 2b. Load FINE-TUNED Policy (Same arch as pre-trained, 4D Output)
    print("\n--- Loading Fine-tuned Policy (4D) ---")
    policy_finetuned = ALGOS[cfg_policy_pt.name.lower()](
        cfg_policy_pt, policy_pretrained_obs_spec, policy_pretrained_action_spec[("agents", "action")],
        base_env.reward_spec, device=base_env.device
    )
    if os.path.exists(cfg.eval.finetuned_ckpt_path):
        policy_finetuned.load_state_dict(torch.load(cfg.eval.finetuned_ckpt_path, weights_only=True))
    else: raise FileNotFoundError(f"Fine-tuned policy not found at {cfg.eval.finetuned_ckpt_path}")
    policy_finetuned.eval()

    # 2c. Load DELTA Policy (Large Arch, 7D Output)
    print("\n--- Loading Delta Action Model (7D) ---")
    cfg_delta_policy = copy.deepcopy(cfg.algo)
    cfg_delta_policy.domain_adaptation = True
    # cfg_delta_policy.actor_hidden_dim = [512, 256, 128]
    # cfg_delta_policy.critic_hidden_dim = [512, 256, 128]

    delta_obs_dim = 32 + 4 # sim_obs + base_action
    delta_obs_spec = base_env.observation_spec.clone()
    delta_obs_spec[("agents", "observation")] = UnboundedContinuousTensorSpec(shape=(1, delta_obs_dim), device=base_env.device)
    
    delta_action_spec = CompositeSpec({"agents": {"action": UnboundedContinuousTensorSpec((1, 6))}}).expand(base_env.num_envs).to(base_env.device)
    # delta_action_spec = CompositeSpec({"agents": {"action": UnboundedContinuousTensorSpec((1, 7))}}).expand(base_env.num_envs).to(base_env.device)

    delta_policy = ALGOS[cfg_delta_policy.name.lower()](
        cfg_delta_policy, delta_obs_spec, delta_action_spec[("agents", "action")], base_env.reward_spec, device=base_env.device
    )
    if os.path.exists(cfg.eval.delta_ckpt_path):
        delta_policy.load_state_dict(torch.load(cfg.eval.delta_ckpt_path, weights_only=True))
    else: raise FileNotFoundError(f"Delta model not found at {cfg.eval.delta_ckpt_path}")
    delta_policy.eval()

    # --- 3. Setup Evaluation Directories ---
    output_dir = os.getcwd()
    model_name = os.path.splitext(os.path.basename(cfg.eval.finetuned_ckpt_path))[0]
    results_dir = os.path.join(output_dir, f"evaluation_FT_comparison_{model_name}")
    os.makedirs(results_dir, exist_ok=True)
    comparison_dir = os.path.join(results_dir, "comparison_plots"); os.makedirs(comparison_dir, exist_ok=True)
    detailed_dir = os.path.join(results_dir, "detailed_plots"); os.makedirs(detailed_dir, exist_ok=True)
    
    # --- 4. Main Evaluation Loop ---
    all_results = []
    checkpoints_s = cfg.eval.get("checkpoints_s", [1.0, 3.0, 5.0])
    dt = cfg.task.sim.dt

    # Define the target state for error calculations
    b1d = base_env.b1d.squeeze().cpu().numpy(); b3d = np.array([0., 0., 1.])
    b2d = np.cross(b3d, b1d); b2d /= (np.linalg.norm(b2d) + 1e-6)
    b1d_ortho = np.cross(b2d, b3d)
    R_target = np.stack([b1d_ortho, b2d, b3d], axis=-1)
    target_state = {
        "y": base_env.payload_target_pos.squeeze().cpu().numpy(), "y_dot": np.zeros(3),
        "q": base_env.init_q.squeeze().cpu().numpy(), "w": np.zeros(3), "W": np.zeros(3), "R": R_target
    }
    for episode_num in tqdm(range(cfg.eval.num_episodes), desc="Running Evaluation Episodes"):
        seed = episode_num 
        
        # Run Pre-trained + Delta
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        results_pretrained = run_episode(policy_pretrained, delta_policy, base_env, cfg.task)

        # Run Fine-tuned + Delta from the same start
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        results_finetuned = run_episode(policy_finetuned, delta_policy, base_env, cfg.task)

        results_pretrained["errors"] = calculate_errors_at_checkpoints(results_pretrained["states"], target_state, checkpoints_s, dt)
        results_finetuned["errors"] = calculate_errors_at_checkpoints(results_finetuned["states"], target_state, checkpoints_s, dt)
        all_results.append({"pretrained": results_pretrained, "finetuned": results_finetuned})
        
        # --- 5. Plotting ---
        plot_comparison_results(results_pretrained, results_finetuned, target_state, episode_num, comparison_dir)
        # plot_detailed_run(results_pretrained, target_state, episode_num, detailed_dir, "Pre-trained+Delta")
        plot_detailed_run(results_finetuned, target_state, episode_num, detailed_dir, "Fine-tuned+Delta")

    # --- 6. Final Summary Table ---
    generate_summary_table(all_results, checkpoints_s, results_dir)

    simulation_app.close()
    print(f"\nEvaluation complete. Plots saved to {results_dir}")

if __name__ == "__main__":
    main()
