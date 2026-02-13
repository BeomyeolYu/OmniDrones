import os
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
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

# Import math utils for the integral controller
from omni_drones.utils.torch import quat_axis, quaternion_to_rotation_matrix

# ==============================================================================
# ==== 1. PLOTTING & SUMMARY FUNCTIONS =========================================
# ==============================================================================

def plot_deployment_comparison(
    results_pt,
    results_pti, # NEW argument
    results_ft,
    results_fti, # Renamed for consistency
    target_state,
    episode_num,
    save_dir
):
    """
    Plots comparison: PT, PT+Int, FT, FT+Int.
    """
    # Unpack data
    s_pt, a_pt, c_pt = results_pt["states"], results_pt["actions"], results_pt["crashed"]
    s_pti, a_pti, c_pti = results_pti["states"], results_pti["actions"], results_pti["crashed"]
    s_ft, a_ft, c_ft = results_ft["states"], results_ft["actions"], results_ft["crashed"]
    s_fti, a_fti, c_fti = results_fti["states"], results_fti["actions"], results_fti["crashed"]
    
    t_pt = np.arange(len(s_pt)) if len(s_pt) > 0 else np.array([])
    t_pti = np.arange(len(s_pti)) if len(s_pti) > 0 else np.array([])
    t_ft = np.arange(len(s_ft)) if len(s_ft) > 0 else np.array([])
    t_fti = np.arange(len(s_fti)) if len(s_fti) > 0 else np.array([])

    # --- Plot 1: Actions ---
    fig_act, axs_act = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig_act.suptitle(f'Action Comparison - Episode {episode_num}', fontsize=16)
    action_labels = ['Thrust (f) [N]', 'Moment (M_x)', 'Moment (M_y)', 'Moment (M_z)']
    
    min_thrust = results_pt.get("min_thrust", 2.0)
    max_thrust = results_pt.get("max_thrust", 40.0)
    max_torque = results_pt.get("max_torque", 0.5)

    def denorm_thrust(norm_val): return (norm_val + 1) / 2 * (max_thrust - min_thrust) + min_thrust

    for i in range(4):
        d_pt = denorm_thrust(a_pt[:, i]) if i==0 else a_pt[:, i] * max_torque
        d_pti = denorm_thrust(a_pti[:, i]) if i==0 else a_pti[:, i] * max_torque
        d_ft = denorm_thrust(a_ft[:, i]) if i==0 else a_ft[:, i] * max_torque
        d_fti = denorm_thrust(a_fti[:, i]) if i==0 else a_fti[:, i] * max_torque

        if len(t_pt) > 0: axs_act[i].plot(t_pt, d_pt, color='blue', linestyle='--', alpha=0.8, label='Pre-trained')
        if len(t_pti) > 0: axs_act[i].plot(t_pti, d_pti, color='cyan', linestyle='-', linewidth=1.5, alpha=0.8, label='PT + Integral')
        if len(t_ft) > 0: axs_act[i].plot(t_ft, d_ft, color='green', linestyle='--', alpha=1.0, label='Fine-tuned')
        if len(t_fti) > 0: axs_act[i].plot(t_fti, d_fti, color='red', linestyle='-', linewidth=1.5, label='FT + Integral')

        # Crashes
        if c_pt and len(t_pt)>0: axs_act[i].scatter(t_pt[-1], d_pt[-1], c='blue', marker='x', s=100, zorder=5)
        if c_pti and len(t_pti)>0: axs_act[i].scatter(t_pti[-1], d_pti[-1], c='cyan', marker='x', s=100, zorder=5)
        if c_ft and len(t_ft)>0: axs_act[i].scatter(t_ft[-1], d_ft[-1], c='green', marker='x', s=100, zorder=5)
        if c_fti and len(t_fti)>0: axs_act[i].scatter(t_fti[-1], d_fti[-1], c='red', marker='x', s=100, zorder=5)

        axs_act[i].set_ylabel(action_labels[i])
        axs_act[i].legend(loc='upper right', fontsize='small')
        axs_act[i].grid(True, linestyle=':')

    axs_act[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(save_dir, f"episode_{episode_num}_actions.png"))
    plt.close(fig_act)

    # --- Plot 2: States ---
    state_plots = {
        "Payload_Pos_Vel": (2, 3, [("Position", "y", "[m]"), ("Velocity", "y_dot", "[m/s]")]),
        "Angles": (3, 3, [("Link Dir", "q", ""), ("Link Ang Vel", "w", "[rad/s]"), ("Drone Ang Vel", "W", "[rad/s]")]),
        "Rotation": (3, 3, [("R_1x", "R", ""), ("R_2x", "R", ""), ("R_3x", "R", "")])
    }
    
    for title, (rows, cols, subplots) in state_plots.items():
        fig, axs = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)
        fig.suptitle(f'{title.replace("_", " ")} - Episode {episode_num}', fontsize=16)
        
        for r in range(rows):
            for c in range(cols):
                ax = axs[r, c] if rows > 1 else axs[c]
                
                if "R_" in subplots[r][0]:
                    idx = 12 + (c * 3 + r) 
                    name = f"R_{c+1}{r+1}"
                    tgt = target_state["R"][c, r]
                else:
                    lbl, key, unit = subplots[r]
                    base = {"y":0, "y_dot":3, "q":6, "w":9, "W":21}[key]
                    idx = base + c
                    name = f"{lbl} {['X','Y','Z'][c]} {unit}"
                    tgt = target_state[key][c]

                ax.axhline(y=tgt, color='k', linestyle=':', alpha=0.6, label='Target')
                if len(t_pt) > 0: ax.plot(t_pt, s_pt[:, idx], 'b--', alpha=0.8, label='Pre-trained')
                if len(t_pti) > 0: ax.plot(t_pti, s_pti[:, idx], 'c-', linewidth=1.5, alpha=0.8, label='PT + Integral')
                if len(t_ft) > 0: ax.plot(t_ft, s_ft[:, idx], 'g--', alpha=1.0, label='Fine-tuned')
                if len(t_fti) > 0: ax.plot(t_fti, s_fti[:, idx], 'r-', linewidth=1.5, label='FT + Integral')
                
                # Crashes
                if c_pt and len(t_pt)>0: ax.scatter(t_pt[-1], s_pt[-1, idx], c='blue', marker='x', s=100, zorder=5)
                if c_pti and len(t_pti)>0: ax.scatter(t_pti[-1], s_pti[-1, idx], c='cyan', marker='x', s=100, zorder=5)
                if c_ft and len(t_ft)>0: ax.scatter(t_ft[-1], s_ft[-1, idx], c='green', marker='x', s=100, zorder=5)
                if c_fti and len(t_fti)>0: ax.scatter(t_fti[-1], s_fti[-1, idx], c='red', marker='x', s=100, zorder=5)

                ax.set_title(name)
                ax.legend(loc='best', fontsize='small')
                ax.grid(True, linestyle=':')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(save_dir, f"episode_{episode_num}_{title.lower()}.png"))
        plt.close(fig)
    
    # # --- Plot 3: Integral States (NEW) ---
    # # Check if integral errors exist for FT+Int (priority) or PT+Int
    # int_err_data = None
    # time_data = None
    # label_prefix = ""
    
    # if results_fti.get("integral_errors") is not None and len(results_fti["integral_errors"]) > 0:
    #     int_err_data = results_fti["integral_errors"]
    #     time_data = t_fti
    #     label_prefix = "FT + Integral"
    # elif results_pti.get("integral_errors") is not None and len(results_pti["integral_errors"]) > 0:
    #     int_err_data = results_pti["integral_errors"]
    #     time_data = t_pti
    #     label_prefix = "PT + Integral"

    # if int_err_data is not None:
    #     fig_int, axs_int = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    #     fig_int.suptitle(f'Integral States (Outer Loop) - Episode {episode_num}', fontsize=16)
        
    #     # u_int is [T, 3], tau_int is [T, 1]
    #     u_x = int_err_data[:, 0]
    #     u_y = int_err_data[:, 1]
    #     u_z = int_err_data[:, 2]
    #     tau = int_err_data[:, 3]
        
    #     labels = ['Pos Int X (u_x)', 'Pos Int Y (u_y)', 'Pos Int Z (u_z)', 'Heading Int (tau)']
    #     data_list = [u_x, u_y, u_z, tau]
        
    #     for i in range(4):
    #         axs_int[i].plot(time_data, data_list[i], 'r-', label=label_prefix)
    #         axs_int[i].set_ylabel(labels[i])
    #         axs_int[i].grid(True, linestyle=':')
    #         axs_int[i].legend(loc='upper right')
            
    #     axs_int[-1].set_xlabel("Timestep")
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    #     plt.savefig(os.path.join(save_dir, f"episode_{episode_num}_integral_states.png"))
    #     plt.close(fig_int)


def calculate_errors(sim_states, target_state, checkpoints_s, dt):
    """Calculates avg errors up to checkpoints + Heading Error."""
    results = {}
    if len(sim_states) == 0: return results

    for t_s in checkpoints_s:
        step = int(t_s / dt)
        if step > len(sim_states): continue
        slc = sim_states[:step]
        if len(slc) == 0: continue

        N = len(slc)
        t_y = np.tile(target_state["y"], (N, 1))
        t_v = np.tile(target_state["y_dot"], (N, 1))
        t_q = np.tile(target_state["q"], (N, 1))
        t_b1 = np.tile(target_state["b1d"], (N, 1))

        errs = {}
        errs["Pos (cm)"] = np.mean(np.linalg.norm(slc[:, 0:3] - t_y, axis=1)) * 100
        errs["Vel (cm/s)"] = np.mean(np.linalg.norm(slc[:, 3:6] - t_v, axis=1)) * 100
        
        dot_q = np.clip(np.einsum('ij,ij->i', slc[:, 6:9], t_q), -1.0, 1.0)
        errs["Link Ang (deg)"] = np.mean(np.degrees(np.arccos(dot_q)))

        R_flat = slc[:, 12:21]
        R = R_flat.reshape(-1, 3, 3).transpose(0, 2, 1)
        b1_curr = R[:, :, 0]
        dot_b1 = np.clip(np.einsum('ij,ij->i', b1_curr, t_b1), -1.0, 1.0)
        errs["Heading (deg)"] = np.mean(np.degrees(np.arccos(dot_b1)))

        results[t_s] = errs
    return results

def generate_summary_table(all_results, checkpoints_s, save_dir):
    if not all_results: return

    lines = ["===== Evaluation Summary ====="]
    lines.append("\n" + "="*40 + "\n          AGGREGATED RESULTS\n" + "="*40)
    
    N = len(all_results)
    succ_pt = sum(1 for r in all_results if not r["pt"]["crashed"])
    succ_pti = sum(1 for r in all_results if not r["pti"]["crashed"])
    succ_ft = sum(1 for r in all_results if not r["ft"]["crashed"])
    # FIX: Key was incorrect ('fti' vs 'fti')
    succ_fti = sum(1 for r in all_results if not r["fti"]["crashed"])
    
    lines.append(f"Pre-trained Success: {succ_pt}/{N} ({succ_pt/N:.1%})")
    lines.append(f"Fine-tuned  Success: {succ_ft}/{N} ({succ_ft/N:.1%})")
    lines.append(f"FT + Integ  Success: {succ_fti}/{N} ({succ_fti/N:.1%})\n")

    valid = [r for r in all_results if not r["pt"]["crashed"]]
    N_stats = len(valid)
    lines.append(f"Stats calculated on {N_stats} episodes where Pre-trained model survived.")
    
    if N_stats > 0:
        sample_err = valid[0]["pt"]["errors"]
        first_key = list(sample_err.keys())[0]
        metrics = list(sample_err[first_key].keys())

        table_data = []
        for t in checkpoints_s:
            row_pt = [f"Pre-trained @ {t}s"]
            row_ft = [f"Fine-tuned @ {t}s"]
            row_fti = [f"FT + Integral @ {t}s"]
            
            for m in metrics:
                v_pt = [r["pt"]["errors"][t][m] for r in valid if t in r["pt"]["errors"]]
                v_ft = [r["ft"]["errors"][t][m] for r in valid if t in r["ft"]["errors"]]
                # FIX: Key was incorrect ('fti' vs 'fti')
                v_fti = [r["fti"]["errors"][t][m] for r in valid if t in r["fti"]["errors"]]

                row_pt.append(f"{np.mean(v_pt):.2f} ± {np.std(v_pt):.2f}" if v_pt else "-")
                row_ft.append(f"{np.mean(v_ft):.2f} ± {np.std(v_ft):.2f}" if v_ft else "-")
                row_fti.append(f"{np.mean(v_fti):.2f} ± {np.std(v_fti):.2f}" if v_fti else "-")
            
            table_data.append(row_pt)
            table_data.append(row_ft)
            table_data.append(row_fti)
            table_data.append(["-" * 12] * (len(metrics) + 1))

        df = pd.DataFrame(table_data, columns=["Model/Time"] + metrics)
        lines.append(df.to_string(index=False))

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(save_dir, "summary_report.txt"), "w") as f: f.write(report)

# ==============================================================================
# ==== 2. LOAD & RUN FUNCTIONS =================================================
# ==============================================================================

def load_policy(cfg, env, path_key):
    from omni_drones.learning import ALGOS
    print(f"--- Loading Policy from cfg.real_world_deployment.{path_key} ---")
    obs_spec = env.observation_spec.clone()
    act_spec = env.action_spec.clone()
    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, obs_spec, act_spec, env.reward_spec, device=env.device
    )
    path = cfg.real_world_deployment.get(path_key, None)
    if path and os.path.exists(path):
        print(f"Loading weights from: {path}")
        state_dict = torch.load(path, weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items(): new_state_dict[k] = v
        policy.load_state_dict(new_state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return policy.eval()

def run_episode(policy, env, integral_gains=None):
    """
    Runs one full episode. 
    If integral_gains=[gamma_pos, gamma_heading] is provided, 
    adds an outer-loop integral controller with HARDCODED MAPPING.
    """
    hist = {"states": [], "actions": [], "integral_errors": []}
    min_t, max_t = env.drone.get_thrust_limits()
    hist["min_thrust"] = min_t.cpu().numpy()[0][0]
    hist["max_thrust"] = max_t.cpu().numpy()[0][0]
    hist["max_torque"] = getattr(env.drone, 'max_torque', 2.0)

    td = env.reset()
    crashed = False
    dt = env.base_env.dt

    # --- Initialize Integrators ---
    u_int = torch.zeros(3, device=env.device) 
    tau_int = torch.zeros(1, device=env.device)

    gamma_pos = 0.0
    gamma_heading = 0.0
    
    # --- HARDCODED SIGNS ---
    # Position Sign: -1.0 (to fix oscillation)
    # Heading Sign: 1.0 (was working)
    s_pos = -1.0
    s_head = 1.0
    
    if integral_gains:
        gamma_pos = integral_gains[0]
        gamma_heading = integral_gains[1]
        print(f"Running with Integral Control: Gamma_pos={gamma_pos} (sign {s_pos}), Gamma_head={gamma_heading} (sign {s_head})")
    
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        for t in range(env.max_episode_length):
            action = policy(td)[("agents", "action")]
            
            if integral_gains:
                # --- Robust State Extraction ---
                drone_state_full = env.base_env.drone_state.reshape(env.base_env.num_envs, -1)[0]
                quat = drone_state_full[3:7].unsqueeze(0)
                
                current_pos = env.base_env.payload_pos.reshape(-1, 3).squeeze(0)
                target_pos = env.base_env.payload_target_pos.reshape(-1, 3).squeeze(0)
                
                # --- Position Integration ---
                err_pos = current_pos - target_pos
                u_int = u_int - gamma_pos * err_pos * dt * s_pos
                
                # Rotate to Body Frame: U = R^T @ u
                R = quaternion_to_rotation_matrix(quat).squeeze(0) # [3, 3]
                U = torch.matmul(R.t(), u_int) # [3]

                # --- Heading Integration ---
                b1 = quat_axis(quat, 0).squeeze(0)
                curr_yaw = torch.atan2(b1[1], b1[0])
                target_b1 = env.base_env.b1d.reshape(-1, 3).squeeze(0)
                targ_yaw = torch.atan2(target_b1[1], target_b1[0])
                
                err_yaw = curr_yaw - targ_yaw
                err_yaw = (err_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                
                tau_int = tau_int - gamma_heading * err_yaw * dt * s_head

                # --- Augment Action (HARDCODED MAPPING) ---
                delta_action = torch.zeros_like(action)
                
                # Thrust = -U[2] (You requested negative U2 for thrust? Note: U[2] is Z)
                # If you meant U_3 from advisor, that is Z, which is index 2 in 0-based
                # Assuming you want: delta_f = -U[2]
                delta_action[..., 0] = -U[2]
                
                # M1 (Action[1]) = -U2 (U[1])
                delta_action[..., 1] = -U[1]
                
                # M2 (Action[2]) = U1 (U[0])
                delta_action[..., 2] = U[0]
                
                # M3 (Action[3]) = tau
                delta_action[..., 3] = tau_int

                action = torch.clamp(action + delta_action, -1.0, 1.0)
                
                hist["integral_errors"].append(torch.cat([u_int, tau_int]).cpu().numpy())
            
            td.set(("agents", "action"), action)
            td = env.step(td)
            
            hist["states"].append(env.get_state())
            hist["actions"].append(action.squeeze().cpu().numpy())

            if td[("next", "done")].any():
                crashed = t < (env.max_episode_length - 1)
                break
            
            td = td["next"]
            
    hist["states"] = np.array(hist["states"])
    hist["actions"] = np.array(hist["actions"])
    if integral_gains:
        hist["integral_errors"] = np.array(hist["integral_errors"])
    hist["crashed"] = crashed
    return hist

# ==============================================================================
# ==== 3. MAIN =================================================================
# ==============================================================================

@hydra.main(version_base=None, config_path=".", config_name="train_f450")
def main(cfg):
    OmegaConf.resolve(cfg); OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs.isaac_env import IsaacEnv
    
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=True)
    env = TransformedEnv(base_env, Compose(InitTracker())).train()
    
    # 1. Load Policies
    policy_pt = load_policy(cfg, env, "pretrained_ckpt_path")
    policy_ft = load_policy(cfg, env, "finetuned_ckpt_path")
    
    # 2. Hardcoded Gains from Request
    int_gains = cfg.real_world_deployment.get("integral_gains", [0.0, 0.0])
    
    print(f"--- Integral Test Enabled (Hardcoded) ---")
    print(f"Gains: {int_gains}")
    print(f"Signs: [-1.0, 1.0]")
    print(f"Mapping: delta_f=-U3(-U[2]), delta_M1=-U2(-U[1]), delta_M2=U1(U[0])")

    # 3. Setup Output
    output_dir = os.getcwd()
    model_name = os.path.splitext(os.path.basename(cfg.real_world_deployment.finetuned_ckpt_path))[0]
    
    gain_str = f"{int_gains[0]:.2f}_{int_gains[1]:.2f}".replace('.', 'p')
    results_dir = os.path.join(output_dir, f"eval_Integral_{gain_str}_Hardcoded_{model_name}")
    os.makedirs(results_dir, exist_ok=True)
    plot_dir = os.path.join(results_dir, "plots"); os.makedirs(plot_dir, exist_ok=True)
    
    # 4. Run Evaluation
    all_results = []
    checkpoints_s = cfg.real_world_deployment.get("checkpoints_s", [32.0])
    dt = cfg.task.sim.dt
    
    print(f"\nStarting Evaluation: {cfg.real_world_deployment.num_episodes} Episodes")

    for i in tqdm(range(cfg.real_world_deployment.num_episodes)):
        seed = i
        
        # --- 1. Pre-trained (Baseline) ---
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); env.set_seed(seed)
        res_pt = run_episode(policy_pt, env, integral_gains=None)

        # --- 2. Pre-trained + Integral (Ablation Test) ---
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); env.set_seed(seed)
        # Using the same hardcoded gains for the PT ablation
        res_pti = run_episode(policy_pt, env, integral_gains=int_gains)

        # --- 3. Fine-tuned (Standard) ---
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); env.set_seed(seed)
        res_ft = run_episode(policy_ft, env, integral_gains=None)

        # --- 4. Fine-tuned + Integral (Target) ---
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); env.set_seed(seed)
        res_fti = run_episode(policy_ft, env, integral_gains=int_gains)

        # Error Calc
        b1d = base_env.b1d.squeeze().cpu().numpy(); b3d = np.array([0., 0., 1.])
        b2d = np.cross(b3d, b1d); b2d /= (np.linalg.norm(b2d) + 1e-6)
        b1d_ortho = np.cross(b2d, b3d)
        R_target = np.stack([b1d_ortho, b2d, b3d], axis=-1)
        target_state = {
            "y": base_env.payload_target_pos.squeeze().cpu().numpy(), "y_dot": np.zeros(3),
            "q": base_env.init_q.squeeze().cpu().numpy(), "w": np.zeros(3), "W": np.zeros(3), 
            "R": R_target, "b1d": b1d
        }
        
        res_pt["errors"] = calculate_errors(res_pt["states"], target_state, checkpoints_s, dt)
        res_pti["errors"] = calculate_errors(res_pti["states"], target_state, checkpoints_s, dt)
        res_ft["errors"] = calculate_errors(res_ft["states"], target_state, checkpoints_s, dt)
        res_fti["errors"] = calculate_errors(res_fti["states"], target_state, checkpoints_s, dt)

        all_results.append({
            "traj_idx": i,
            "pt": res_pt,
            "pti": res_pti,
            "ft": res_ft,
            "fti": res_fti
        })

        plot_deployment_comparison(res_pt, res_pti, res_ft, res_fti, target_state, i, plot_dir)

    # 5. Final Report
    generate_summary_table(all_results, checkpoints_s, results_dir)
    
    simulation_app.close()
    print(f"Done. Results saved to {results_dir}")

if __name__ == "__main__":
    main()