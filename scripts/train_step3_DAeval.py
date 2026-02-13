import os
import hydra
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict import TensorDict

# ==============================================================================
# ==== 1. ALL PLOTTING AND HELPER FUNCTIONS ====================================
# ==============================================================================

def plot_evaluation_results(
    real_action_hist,
    delta_action_hist, # This will now be 7D
    corrected_action_hist,
    real_next_state_hist,
    sim_next_state_hist,
    traj_idx,
    save_dir,
    mode="ASAP",
    min_thrust=None,
    max_thrust=None,
    drone_mass=None,
    gravity_ratio=None
):
    """
    Plots the detailed, multi-panel comparison for a single run.
    Now handles 7D delta actions and plots fictitious forces.
    """
    delta_actions_7d = np.array(delta_action_hist)
    corrected_actions_4d = np.array(corrected_action_hist)
    sim_next_states = np.array(sim_next_state_hist)
    
    real_actions_4d = np.array(real_action_hist)
    real_next_states = np.array(real_next_state_hist)

    sim_len = len(sim_next_states)
    full_len = len(real_next_states)
    crashed = sim_len < full_len

    time_sim = np.arange(sim_len)
    time_full = np.arange(full_len)

    # --- Plot 1: Control Actions (4D) ---
    fig_act, axs_act = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    fig_act.suptitle(f'Detailed Control Actions ({mode}) - Trajectory {traj_idx}', fontsize=16)
    action_labels = ['Total Thrust (f) [N]', 'Moment (M_x) [Nm]', 'Moment (M_y) [Nm]', 'Moment (M_z) [Nm]']
    
    for i in range(4):
        delta_controls = delta_actions_7d[:, i] if delta_actions_7d.shape[0] > 0 else []
        
        # --- Denormalize the thrust component (i=0) ---
        if i == 0 and min_thrust is not None and max_thrust is not None:
            # Denormalization function for a [-1, 1] range
            def denorm(norm_val):
                return (norm_val + 1) / 2 * (max_thrust - min_thrust) + min_thrust

            real_thrust_N = denorm(real_actions_4d[:, i])
            corrected_thrust_N = denorm(corrected_actions_4d[:, i])
            # The delta is the difference in physical Newtons
            delta_thrust_N = corrected_thrust_N - real_thrust_N[:len(corrected_thrust_N)]

            axs_act[i].plot(time_full, real_thrust_N, 'b--', alpha=0.5, label='Recorded Real Thrust [N]')
            axs_act[i].plot(time_sim, delta_thrust_N, 'g-', label='Delta Thrust [N]')
            axs_act[i].plot(time_sim, corrected_thrust_N, 'r-', label='Corrected Thrust [N]')
        else:
            # --- Denormalize the moments (i > 0) ---
            max_moment = 2.
            if i > 0 and max_moment is not None:
                real_moment = real_actions_4d[:, i] * max_moment
                corrected_moment = corrected_actions_4d[:, i] * max_moment
                delta_moment = corrected_moment - real_moment[:len(corrected_moment)]

                axs_act[i].plot(time_full, real_moment, 'b--', alpha=0.5, label='Recorded Real Moment [N·m]')
                axs_act[i].plot(time_sim, delta_moment, 'g-', label='Delta Moment [N·m]')
                axs_act[i].plot(time_sim, corrected_moment, 'r-', label='Corrected Moment [N·m]')
            else:
                # Fallback to plotting normalized values if no scale is provided
                axs_act[i].plot(time_full, real_actions_4d[:, i], 'b--', alpha=0.5, label='Recorded Real Action')
                axs_act[i].plot(time_sim, delta_controls, 'g-', label='Delta Control (Δa_c)')
                axs_act[i].plot(time_sim, corrected_actions_4d[:, i], 'r-', label='Corrected Action')
        # -------------------------------------------------------------

        axs_act[i].set_ylabel(action_labels[i])
        if crashed:
            # This logic needs to handle both cases
            value_to_plot = corrected_thrust_N[-1] if i == 0 and min_thrust is not None else corrected_actions_4d[-1, i]
            axs_act[i].scatter(time_sim[-1], value_to_plot, color='red', s=40, zorder=5, label='Crash Point')
        
        axs_act[i].legend(); axs_act[i].grid(True, linestyle=':')
    
    axs_act[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(save_dir, f"traj{traj_idx}_{mode}_1_control_acts.png"))
    plt.close(fig_act)

    # --- Fictitious Forces (3D) ---
    fig_force, axs_force = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    fig_force.suptitle(f'Learned Fictitious Forces ({mode}) - Trajectory {traj_idx}', fontsize=16)
    force_labels = ['Fictitious Force (a_x) [N]', 'Fictitious Force (a_y) [N]', 'Fictitious Force (a_z) [N]']
    
    for i in range(2): #for i in range(3):
        fict_force_raw = delta_actions_7d[:, 4 + i] if delta_actions_7d.shape[0] > 0 else []
        
        # --- THE FIX: Calculate the physical force in Newtons ---
        scaled_fict_force = fict_force_raw # Default to raw if no scaling info
        if drone_mass is not None and gravity_ratio is not None:
            max_fictitious_force = drone_mass * 9.81 * gravity_ratio
            scaled_fict_force = fict_force_raw * max_fictitious_force
        # ------------------------------------------------------

        axs_force[i].plot(time_sim, scaled_fict_force, 'm-', label='Applied Force [N]') # Use the scaled value
        axs_force[i].set_ylabel(force_labels[i])
        if crashed:
            axs_force[i].scatter(time_sim[-1], scaled_fict_force[-1], color='red', s=40, zorder=5, label='Crash Point')
        axs_force[i].legend(); axs_force[i].grid(True, linestyle=':')
    
    axs_force[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # Note: I'm updating the filename to avoid overwriting your old plots
    plt.savefig(os.path.join(save_dir, f"traj{traj_idx}_{mode}_2_fictitious_forces.png"))
    plt.close(fig_force)

    # --- Plot 2: Payload Position (y) and Velocity (y_dot) ---
    fig_y, axs_y = plt.subplots(2, 3, figsize=(15, 8))
    fig_y.suptitle(f'Detailed Payload State ({mode}) - Trajectory {traj_idx}', fontsize=16)
    for i in range(3):
        axis_label = ['1', '2', '3'][i]
        axs_y[0, i].plot(time_full, real_next_states[:, i], 'b--', label='Real Next')
        axs_y[1, i].plot(time_full, real_next_states[:, 3+i], 'b--', label='Real Next')
        axs_y[0, i].plot(time_sim, sim_next_states[:, i], 'r-', label='Sim Next')
        axs_y[1, i].plot(time_sim, sim_next_states[:, 3+i], 'r-', label='Sim Next')
        if crashed:
            axs_y[0, i].scatter(time_sim[-1], sim_next_states[-1, i], color='red', s=40, alpha=0.5, zorder=5, label='Crash Point')
            axs_y[1, i].scatter(time_sim[-1], sim_next_states[-1, 3+i], color='red', s=40, alpha=0.5, zorder=5)
        axs_y[0, i].set_title(f'Payload Position, y_{axis_label} [m]'); axs_y[0, i].legend(); axs_y[0, i].grid(True, linestyle=':')
        axs_y[1, i].set_title(f'Payload Velocity, ydot_{axis_label} [m/s]'); axs_y[1, i].legend(); axs_y[1, i].grid(True, linestyle=':')
    fig_y.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(save_dir, f"traj{traj_idx}_{mode}_3_y_ydot.png"))
    plt.close(fig_y)

    # --- Plot 3: Bar (q), Bar Velocity (w), Drone Velocity (W) ---
    fig_q, axs_q = plt.subplots(3, 3, figsize=(15, 10))
    fig_q.suptitle(f'Detailed Angular States ({mode}) - Trajectory {traj_idx}', fontsize=16)
    for i in range(3):
        axis_label = ['1', '2', '3'][i]
        axs_q[0, i].plot(time_full, real_next_states[:, 6+i], 'b--', label='Real Next')
        axs_q[1, i].plot(time_full, real_next_states[:, 9+i], 'b--', label='Real Next')
        axs_q[2, i].plot(time_full, real_next_states[:, 21+i], 'b--', label='Real Next')
        axs_q[0, i].plot(time_sim, sim_next_states[:, 6+i], 'r-', label='Sim Next')
        axs_q[1, i].plot(time_sim, sim_next_states[:, 9+i], 'r-', label='Sim Next')
        axs_q[2, i].plot(time_sim, sim_next_states[:, 21+i], 'r-', label='Sim Next')
        if crashed:
            axs_q[0, i].scatter(time_sim[-1], sim_next_states[-1, 6+i], color='red', s=40, alpha=0.5, zorder=5, label='Crash Point')
            axs_q[1, i].scatter(time_sim[-1], sim_next_states[-1, 9+i], color='red', s=40, alpha=0.5, zorder=5)
            axs_q[2, i].scatter(time_sim[-1], sim_next_states[-1, 21+i], color='red', s=40, alpha=0.5, zorder=5)
        axs_q[0, i].set_title(f'Bar Direction, q_{axis_label}'); axs_q[0, i].legend(); axs_q[0, i].grid(True, linestyle=':')
        axs_q[1, i].set_title(f'Bar Ang. Vel., w_{axis_label} [rad/s]'); axs_q[1, i].legend(); axs_q[1, i].grid(True, linestyle=':')
        axs_q[2, i].set_title(f'Drone Ang. Vel., W_{axis_label} [rad/s]'); axs_q[2, i].legend(); axs_q[2, i].grid(True, linestyle=':')
    fig_q.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(save_dir, f"traj{traj_idx}_{mode}_4_q_w_W.png"))
    plt.close(fig_q)

    # --- Plot 4: Rotation Matrix (R) ---
    fig_r, axs_r = plt.subplots(3, 3, figsize=(15, 10))
    fig_r.suptitle(f'Detailed Rotation Matrix ({mode}) - Trajectory {traj_idx}', fontsize=16)
    for i in range(3):
        for j in range(3):
            idx = 3 * i + j
            axs_r[i, j].plot(time_full, real_next_states[:, 12 + idx], 'b--', label='Real Next')
            axs_r[i, j].plot(time_sim, sim_next_states[:, 12 + idx], 'r-', label='Sim Next')
            if crashed:
                axs_r[i, j].scatter(time_sim[-1], sim_next_states[-1, 12 + idx], color='red', s=40, alpha=0.5, zorder=5, label='Crash Point')
            axs_r[i, j].set_title(f'R_{i+1}{j+1}'); axs_r[i, j].legend(); axs_r[i, j].grid(True, linestyle=':')
    fig_r.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(save_dir, f"traj{traj_idx}_{mode}_5_R.png"))
    plt.close(fig_r)

def plot_trajectory_comparison(
    vanilla_sim_states, asap_sim_states, real_states,
    traj_idx, save_dir, dt
):
    """
    Plots the comparison of position error between Vanilla (open-loop) and
    ASAP (delta action) for a single trajectory.
    """
    len_vanilla = len(vanilla_sim_states)
    len_asap = len(asap_sim_states)
    
    time_vanilla = np.arange(len_vanilla) * dt
    time_asap = np.arange(len_asap) * dt

    vanilla_pos_error = np.linalg.norm(vanilla_sim_states[:, 0:3] - real_states[:len_vanilla, 0:3], axis=1) * 100
    asap_pos_error = np.linalg.norm(asap_sim_states[:, 0:3] - real_states[:len_asap, 0:3], axis=1) * 100

    plt.figure(figsize=(12, 6))
    plt.plot(time_vanilla, vanilla_pos_error, label='Vanilla (Open-Loop Action Only)', color='blue')
    plt.plot(time_asap, asap_pos_error, label='ASAP (Open-Loop + Delta Action)', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error, $e_y$ (cm)')
    plt.title(f'Position Tracking Error Comparison - Trajectory {traj_idx}')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"traj{traj_idx}_pos_error_comparison.png"))
    plt.close()

def vee(mat_batch):
    """Numpy implementation of the vee operator for a batch of 3x3 matrices."""
    return np.stack([mat_batch[..., 2, 1], mat_batch[..., 0, 2], mat_batch[..., 1, 0]], axis=-1)

# ==============================================================================
# ==== 2. ERROR CALCULATION with GEOMETRIC/LIE ERRORS ==========================
# ==============================================================================

def calculate_errors_at_checkpoints(
    sim_next_state_hist,
    real_next_state_hist,
    checkpoints_s,
    dt
):
    """
    Calculates error metrics at checkpoints using the same geometric and Lie algebra
    formulations as the training environment.
    """
    sim_states = np.array(sim_next_state_hist)
    real_states = np.array(real_next_state_hist)
    
    checkpoint_steps = [int(s / dt) for s in checkpoints_s]
    results = {}

    for i, step_limit in enumerate(checkpoint_steps):
        time_s = checkpoints_s[i]
        
        num_steps = min(step_limit, len(sim_states), len(real_states))
        if num_steps == 0: continue

        sim_slice = sim_states[:num_steps]
        real_slice = real_states[:num_steps]
        
        # E_pos (cm): Standard RMSE on payload position
        pos_error = np.linalg.norm(sim_slice[:, 0:3] - real_slice[:, 0:3], axis=1)
        mean_pos_error_cm = np.mean(pos_error) * 100

        # E_vel (cm/s): Standard RMSE on payload velocity
        vel_error = np.linalg.norm(sim_slice[:, 3:6] - real_slice[:, 3:6], axis=1)
        mean_vel_error_cms = np.mean(vel_error) * 100

        # E_b1 (deg): Heading error, angle between the drone's forward vectors (x-axis)
        R_sim_h = sim_slice[:, 12:21].reshape(-1, 3, 3)
        R_real_h = real_slice[:, 12:21].reshape(-1, 3, 3)
        b1_sim = R_sim_h[:, :, 0]
        b1_real = R_real_h[:, :, 0]
        dot_h = np.einsum('ij,ij->i', b1_sim, b1_real)
        angles_h = np.arccos(np.clip(dot_h, -1.0, 1.0))
        mean_heading_error_deg = np.mean(np.degrees(angles_h))
        
        # E_q (deg): Bar direction error, angle between q vectors
        q_sim = sim_slice[:, 6:9]
        q_real = real_slice[:, 6:9]
        dot_q = np.einsum('ij,ij->i', q_sim, q_real)
        angles_q = np.arccos(np.clip(dot_q, -1.0, 1.0))
        mean_q_error_deg = np.mean(np.degrees(angles_q))
        
        # E_w (rad/s): Geometric error norm for link angular velocity
        w_sim, q_sim, w_real = sim_slice[:, 9:12], sim_slice[:, 6:9], real_slice[:, 9:12]
        ew_vec = w_sim + np.cross(q_sim, np.cross(q_sim, w_real, axis=1), axis=1)
        mean_w_error_rads = np.mean(np.linalg.norm(ew_vec, axis=1))

        # E_R (rad): Lie algebra error norm for drone orientation
        R_sim = sim_slice[:, 12:21].reshape(-1, 3, 3)
        R_real = real_slice[:, 12:21].reshape(-1, 3, 3)
        R_sim_T = R_sim.transpose(0, 2, 1)
        R_real_T = real_slice[:, 12:21].reshape(-1, 3, 3).transpose(0, 2, 1)
        # eR = 0.5 * vee(R_d^T @ R - R^T @ R_d)
        skew_symm_batch = np.einsum('...ij,...jk->...ik', R_real_T, R_sim) - np.einsum('...ij,...jk->...ik', R_sim_T, R_real)
        eR_vec = 0.5 * vee(skew_symm_batch)
        mean_R_error_rad = np.mean(np.linalg.norm(eR_vec, axis=1))

        # E_W (rad/s): Geometric error norm for drone angular velocity
        W_sim, W_real = sim_slice[:, 21:24], real_slice[:, 21:24]
        # eW = W_sim - R_sim^T @ R_real @ W_real
        transported_W_real = np.einsum('...ij,...jk,...k->...i', R_sim_T, R_real, W_real)
        eW_vec = W_sim - transported_W_real
        mean_W_error_rads = np.mean(np.linalg.norm(eW_vec, axis=1))

        results[time_s] = {
            "E_pos (cm)": mean_pos_error_cm,
            "E_vel (cm/s)": mean_vel_error_cms,
            "E_b1 (deg)": mean_heading_error_deg,
            "E_q (deg)": mean_q_error_deg,
            "E_w (rad/s)": mean_w_error_rads,
            "E_R (rad)": mean_R_error_rad,
            "E_W (rad/s)": mean_W_error_rads,
        }
        
    return results

# ==============================================================================
# ==== 3. DETAILED SUMMARY TABLE GENERATION ====================================
# ==============================================================================

def generate_summary_table(all_results, checkpoints_s, save_dir, dt):
    """
    Generates a detailed report with per-trajectory tables for Vanilla vs ASAP
    and a final aggregated summary with mean, std dev, and success rate.
    """
    if not all_results:
        print("No results to summarize.")
        return

    report_parts = ["===== Evaluation Summary ====="]
    
    # Part 1: Per-Trajectory Tables
    for result_pair in all_results:
        traj_idx, vanilla_res, asap_res = result_pair['traj_idx'], result_pair['vanilla'], result_pair['asap']
        report_parts.append(f"\n--- Trajectory {traj_idx} ---")
        
        # Vanilla run
        report_parts.append("\n  Vanilla (Open-Loop Action Only):")
        if not vanilla_res['successful']:
            fail_info = vanilla_res['failure_info']
            report_parts.append(f"  Status: crashed at step {fail_info['step']}, {fail_info['time']:.3f}s (payload Z position: {fail_info['z_pos']:.2f}m).")
        else:
            report_parts.append("  Status: SUCCESSFUL")
            df_vanilla_data = {"Length": [f"{s}s" for s in checkpoints_s]}
            metric_names = list(vanilla_res['errors'][checkpoints_s[0]].keys()) if (checkpoints_s and vanilla_res['errors']) else []
            for metric in metric_names:
                df_vanilla_data[metric] = [vanilla_res['errors'][s][metric] for s in checkpoints_s]
            df_vanilla = pd.DataFrame(df_vanilla_data)
            report_parts.append(df_vanilla.to_string(index=False, float_format='%.2f'))

        # ASAP run
        report_parts.append("\n  ASAP (Open-Loop Action + Delta Action):")
        if not asap_res['successful']:
            fail_info = asap_res['failure_info']
            report_parts.append(f"  Status: crashed at step {fail_info['step']}, {fail_info['time']:.3f}s (payload Z position: {fail_info['z_pos']:.2f}m).")
        else:
            report_parts.append("  Status: SUCCESSFUL")
            df_asap_data = {"Length": [f"{s}s" for s in checkpoints_s]}
            metric_names = list(asap_res['errors'][checkpoints_s[0]].keys()) if (checkpoints_s and asap_res['errors']) else []
            for metric in metric_names:
                df_asap_data[metric] = [asap_res['errors'][s][metric] for s in checkpoints_s]
            df_asap = pd.DataFrame(df_asap_data)
            report_parts.append(df_asap.to_string(index=False, float_format='%.2f'))

    # Part 2: Aggregated Summary
    report_parts.append("\n\n" + "="*30 + "\n          AGGREGATED RESULTS\n" + "="*30 + "\n")
    
    # Success Rates
    num_total_trajs = len(all_results)
    num_vanilla_success = sum(1 for res_pair in all_results if res_pair['vanilla']['successful'])
    num_asap_success = sum(1 for res_pair in all_results if res_pair['asap']['successful'])
    
    vanilla_success_rate = (num_vanilla_success / num_total_trajs) * 100 if num_total_trajs > 0 else 0
    asap_success_rate = (num_asap_success / num_total_trajs) * 100 if num_total_trajs > 0 else 0
    
    report_parts.append(f"Vanilla Success Rate: {vanilla_success_rate:.1f}% ({num_vanilla_success}/{num_total_trajs} trajectories)")
    report_parts.append(f"ASAP Success Rate: {asap_success_rate:.1f}% ({num_asap_success}/{num_total_trajs} trajectories)\n")

    # Aggregated Error Metrics (for successful runs only)
    successful_vanilla_results = [res_pair['vanilla'] for res_pair in all_results if res_pair['vanilla']['successful']]
    successful_asap_results = [res_pair['asap'] for res_pair in all_results if res_pair['asap']['successful']]

    if not successful_vanilla_results and not successful_asap_results:
        report_parts.append("\nNo successful trajectories for either method to aggregate.")
    else:
        metric_names = []
        if successful_asap_results: metric_names = list(successful_asap_results[0]['errors'][checkpoints_s[0]].keys())
        elif successful_vanilla_results: metric_names = list(successful_vanilla_results[0]['errors'][checkpoints_s[0]].keys())
        
        agg_table_data = {"Length": [f"{s}s" for s in checkpoints_s]}
        for metric in metric_names:
            agg_table_data[f"Vanilla {metric}"] = []
            agg_table_data[f"ASAP {metric}"] = []
        for time_s in checkpoints_s:
            for metric in metric_names:
                # Vanilla
                vanilla_metrics = [res['errors'][time_s][metric] for res in successful_vanilla_results if time_s in res['errors']]
                if vanilla_metrics:
                    mean_val, std_val = np.mean(vanilla_metrics), np.std(vanilla_metrics)
                    agg_table_data[f"Vanilla {metric}"].append(f"{mean_val:.2f} ± {std_val:.2f}")
                else: agg_table_data[f"Vanilla {metric}"].append("N/A")
                
                # ASAP
                asap_metrics = [res['errors'][time_s][metric] for res in successful_asap_results if time_s in res['errors']]
                if asap_metrics:
                    mean_val, std_val = np.mean(asap_metrics), np.std(asap_metrics)
                    agg_table_data[f"ASAP {metric}"].append(f"{mean_val:.2f} ± {std_val:.2f}")
                else: agg_table_data[f"ASAP {metric}"].append("N/A")
        
        df_agg = pd.DataFrame(agg_table_data)
        report_parts.append("\n" + df_agg.to_string(index=False))
        report_parts.append("\nError metrics are calculated over successful trajectories only.")

    # Print and Save the final report
    final_report = "\n".join(report_parts)
    print("\n" + final_report)
    report_path = os.path.join(save_dir, "evaluation_summary_report.txt")
    with open(report_path, "w") as f: f.write(final_report)
    print(f"\nDetailed summary report saved to {report_path}")

# ==============================================================================
# ==== 4. TRAJECTORY SIMULATION HELPER FUNCTION ================================
# ==============================================================================

def run_trajectory_simulation(
    base_env, policy, real_data, traj_idx, dt,
    failure_height_threshold, mode="ASAP"
):
    """
    Runs a single trajectory simulation, now handling 7D actions.
    """
    # This reset logic is correct.
    base_env.set_real_world_data(real_data, eval_trajectory_idx=traj_idx)
    base_env.reset()
    base_env.sim.step(render=False)

    # --- Get the thrust limits after the env is reset ---
    drone_mass = base_env.drone.masses.cpu().numpy()[0][0][0]
    min_thrust_t, max_thrust_t = base_env.drone.get_thrust_limits()
    # Convert to simple float values for easy use later
    min_thrust = min_thrust_t.cpu().numpy()[0][0]
    max_thrust = max_thrust_t.cpu().numpy()[0][0]
    # -----------------------------------------------------------

    is_successful = True
    failure_info = None

    real_actions_traj = real_data["actions"][traj_idx]
    if real_actions_traj.dtype == 'object':
        real_actions_traj = np.vstack(real_actions_traj).astype(np.float32)

    real_next_states_traj = real_data["next_states"][traj_idx]
    if real_next_states_traj.dtype == 'object':
        real_next_states_traj = np.vstack(real_next_states_traj).astype(np.float32)
    
    sim_next_state_history = []
    real_action_history = []
    delta_action_history = [] # This will now store the full 7D delta action
    corrected_action_history = [] # This will store the 4D corrected *control* action

    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        for t in range(len(real_actions_traj)):
            obs_td = base_env._compute_state_and_obs()
            real_action_t = torch.from_numpy(real_actions_traj[t]).float().to(base_env.device)
            
            if mode == "ASAP" and policy is not None:
                '''
                # Get the full 7D delta action from the policy
                delta_action_7d = policy(obs_td)[("agents", "action")].squeeze()
                
                # Split the 7D action into its components
                delta_controls_4d = delta_action_7d[:4]
                
                # The corrected *control* action is 4D
                corrected_controls_4d = torch.clamp(real_action_t + delta_controls_4d, min=-1., max=1.)
                
                # The action applied to the environment step is the full 7D vector
                action_to_apply = delta_action_7d.reshape(1, 1, -1)

                # Store the actions for plotting
                delta_action_history.append(delta_action_7d.cpu().numpy())
                corrected_action_history.append(corrected_controls_4d.cpu().numpy())        
                '''
                # Get the full 7D delta action from the policy
                delta_action_6d = policy(obs_td)[("agents", "action")].squeeze()
                
                # Split the 7D action into its components
                delta_controls_4d = delta_action_6d[:4]
                
                # The corrected *control* action is 4D
                corrected_controls_4d = torch.clamp(real_action_t + delta_controls_4d, min=-1., max=1.)
                
                # The action applied to the environment step is the full 7D vector
                action_to_apply = delta_action_6d.reshape(1, 1, -1)

                # Store the actions for plotting
                delta_action_history.append(delta_action_6d.cpu().numpy())
                corrected_action_history.append(corrected_controls_4d.cpu().numpy())

            else: # Vanilla mode (no delta action)
                # The action to apply is just the real action (4D)
                # We need to pad it to 7D for the env step if it expects a 7D input now
                '''
                vanilla_action_7d = torch.zeros(7, device=base_env.device)
                vanilla_action_7d[:4] = real_action_t
                action_to_apply = vanilla_action_7d.reshape(1, 1, -1)

                # Store placeholder values for plotting
                delta_action_history.append(torch.zeros(7).numpy())
                corrected_action_history.append(real_action_t.cpu().numpy())
                '''
                vanilla_action_6d = torch.zeros(6, device=base_env.device)
                vanilla_action_6d[:4] = real_action_t
                action_to_apply = vanilla_action_6d.reshape(1, 1, -1)

                # Store placeholder values for plotting
                delta_action_history.append(torch.zeros(6).numpy())
                corrected_action_history.append(real_action_t.cpu().numpy())

            base_env.step(TensorDict({"agents": {"action": action_to_apply}}, batch_size=[1]))
            
            current_sim_state = base_env.get_state()
            sim_next_state_history.append(current_sim_state)
            real_action_history.append(real_action_t.cpu().numpy())
            
            payload_z_pos = current_sim_state[2]
            if payload_z_pos < failure_height_threshold:
                is_successful = False
                failure_info = {"step": t, "time": t * dt, "z_pos": payload_z_pos}
                tqdm.write(f"Trajectory {traj_idx} ({mode}) failed at step {t} (payload Z position: {payload_z_pos:.2f}m).")
                break
    
    if not sim_next_state_history:
        sim_next_state_history = np.zeros_like(real_next_states_traj[0:1])

    return {
        "sim_states": np.array(sim_next_state_history),
        "real_states": real_next_states_traj,
        "successful": is_successful,
        "failure_info": failure_info,
        "real_actions": real_action_history,
        "delta_actions": delta_action_history,
        "corrected_actions": corrected_action_history,
        "min_thrust": min_thrust,
        "max_thrust": max_thrust,
        "drone_mass": drone_mass
    }

# ==============================================================================
# ==== 5. MAIN FUNCTION ========================================================
# ==============================================================================

@hydra.main(version_base=None, config_path=".", config_name="train_f450")
def main(cfg):
    
    #### --- CONFIGURATION FLAG FOR PLOTTING ------------------------------------------------------
    '''
    # real_data_path = "test_dataset_100trajs_1000steps.npz"
    # pretrained_delta_action_model = "checkpoint_best_11100.pth" 
    # pretrained_delta_action_model = "checkpoint_best_16500.pth"
    # pretrained_delta_action_model = "checkpoint_466616320.pth"
    '''
    '''
    real_data_path = "deltaTrainingDataset_200trajs_500steps_chunk_20251006_181736.npz"
    pretrained_delta_action_model = "checkpoint_92798976.pth"
    '''
    '''
    # real_data_path = "delta_act_training_dataset/testTraj_deltaTrainingDataset_100trajs_500steps.npz"
    real_data_path = "delta_act_training_dataset/testTraj_deltaTrainingDataset_100trajs_500steps_chunk_20251010_003000.npz"
    pretrained_delta_action_model = "checkpoint_DA_19600T_186908672.pth"
    '''

    # real_data_path = "TEST_deltaTrainingDataset_100trajs.npz"
    # real_data_path = "TEST_Inv_deltaTrainingDataset_100trajs.npz"
    real_data_path = "TEST_Inv_deltaTrainingDataset_100trajs_130.npz"

    # pretrained_delta_action_model = "checkpoint_100532224.pth"
    # pretrained_delta_action_model = "checkpoint_137101312.pth"
    # pretrained_delta_action_model = "checkpoint_103219200.pth"

    # pretrained_delta_action_model = "checkpoint_120193024.pth" 
    # pretrained_delta_action_model = "checkpoint_110886912.pth"
    # pretrained_delta_action_model = "checkpoint_116916224.pth"

    # pretrained_delta_action_model = "checkpoint_85824000.pth"
    # pretrained_delta_action_model = "checkpoint_111424000.pth" 

    # pretrained_delta_action_model = "checkpoint_268050240_best.pth" 
    pretrained_delta_action_model = "checkpoint_256314240_best.pth" 
    

    # pretrained_delta_action_model = ".pth" 
    # pretrained_delta_action_model = ".pth"
    # pretrained_delta_action_model = ".pth"
    # pretrained_delta_action_model = ".pth" 
    # pretrained_delta_action_model = ".pth"
    

    save_detailed_plots = True # Set to False to disable detailed multi-panel plots
    test_trajectory_indices = list(range(10)) #[0, 1, 2] # test indices
    ''' --- HOW TO SELECT TRAJECTORIES ---
    # To select a range (e.g., 0 to 10), use: list(range(10))
    # To select a slice (e.g., 50 to 100), use: list(range(50, 100))
    # To select all trajectories, first get the total number:
    # num_trajectories = real_data["states"].shape[0] then use: list(range(num_trajectories)) '''

    checkpoints_s = [1., 8.]  #[1., 3., 5.]
    dt = cfg.task.sim.dt
    failure_height_threshold = 0.1
    #### ------------------------------------------------------------------------------------------
    
    # 1. Setup
    OmegaConf.register_new_resolver("eval", eval); OmegaConf.resolve(cfg); OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs.isaac_env import IsaacEnv
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=True)
    env_for_policy_spec = TransformedEnv(base_env, Compose(*[InitTracker()])).eval()
    
    # 2. Load Data
    data_path = os.path.join(os.path.dirname(__file__), real_data_path)
    # real_data = np.load(data_path, allow_pickle=True)
    # --- FIX: Convert the NpzFile object to a standard dictionary ---
    with np.load(data_path, allow_pickle=True) as npz_file:
        real_data = {key: npz_file[key] for key in npz_file.files}
    # -----------------------------------------------------------------
    print(list(range(real_data["states"].shape[0])))

    # 3. Load Policy
    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env_for_policy_spec.observation_spec, env_for_policy_spec.action_spec, env_for_policy_spec.reward_spec, device=base_env.device
    )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if pretrained_delta_action_model:
        pretrained_ckpt_path = os.path.join(base_dir, pretrained_delta_action_model)
        state_dict = torch.load(pretrained_ckpt_path, map_location=base_env.device, weights_only=True)
        policy.load_state_dict(state_dict); print(f"Successfully loaded model from {pretrained_ckpt_path}")
    else: raise ValueError("A `pretrained_ckpt_path` must be provided for evaluation.")

    # 4. Setup Evaluation
    all_comparison_results = []
    output_dir = os.getcwd(); model_name = os.path.splitext(os.path.basename(pretrained_delta_action_model))[0]
    results_dir = os.path.join(output_dir, f"evaluation_DA_comparison_results_{model_name}")
    os.makedirs(results_dir, exist_ok=True)
    plot_comparison_dir = os.path.join(results_dir, "comparison_plots"); os.makedirs(plot_comparison_dir, exist_ok=True)
    detailed_plot_dir = os.path.join(results_dir, "detailed_plots")
    if save_detailed_plots: os.makedirs(detailed_plot_dir, exist_ok=True)

    # 5. Main Evaluation Loop
    for traj_idx in tqdm(test_trajectory_indices, desc="Evaluating All Trajectories"):
        # Vanilla
        # '''
        tqdm.write(f"\nRunning Trajectory {traj_idx} (Vanilla)...")
        vanilla_results = run_trajectory_simulation(base_env, None, real_data, traj_idx, dt, failure_height_threshold, mode="Vanilla")
        vanilla_results['errors'] = calculate_errors_at_checkpoints(vanilla_results['sim_states'], vanilla_results['real_states'], checkpoints_s, dt) if vanilla_results['successful'] else {}
        # '''

        # ASAP
        tqdm.write(f"Running Trajectory {traj_idx} (ASAP)...")
        asap_results = run_trajectory_simulation(base_env, policy, real_data, traj_idx, dt, failure_height_threshold, mode="ASAP")
        asap_results['errors'] = calculate_errors_at_checkpoints(asap_results['sim_states'], asap_results['real_states'], checkpoints_s, dt) if asap_results['successful'] else {}
        
        # Store results
        # '''
        all_comparison_results.append({"traj_idx": traj_idx, "vanilla": vanilla_results, "asap": asap_results})
        # '''

        # Generate plots
        '''
        plot_trajectory_comparison(vanilla_results['sim_states'], asap_results['sim_states'], vanilla_results['real_states'], traj_idx, plot_comparison_dir, dt)
        '''

        if save_detailed_plots:
            full_real_actions = real_data["actions"][traj_idx]
            full_real_states = real_data["next_states"][traj_idx]
            
            '''
            vanilla_plot_args = {
                "real_action_hist": full_real_actions, "delta_action_hist": vanilla_results['delta_actions'],
                "corrected_action_hist": vanilla_results['corrected_actions'], "real_next_state_hist": full_real_states,
                "sim_next_state_hist": vanilla_results['sim_states'],
                "min_thrust": vanilla_results['min_thrust'], "max_thrust": asap_results['max_thrust'],
                "drone_mass": vanilla_results['drone_mass'], "gravity_ratio": cfg.task.fictitious_force_gravity_ratio
            }
            plot_evaluation_results(**vanilla_plot_args, traj_idx=traj_idx, save_dir=detailed_plot_dir, mode="Vanilla")
            '''

            asap_plot_args = {
                "real_action_hist": full_real_actions, "delta_action_hist": asap_results['delta_actions'],
                "corrected_action_hist": asap_results['corrected_actions'], "real_next_state_hist": full_real_states,
                "sim_next_state_hist": asap_results['sim_states'],
                "min_thrust": asap_results['min_thrust'], "max_thrust": asap_results['max_thrust'],
                "drone_mass": asap_results['drone_mass'], "gravity_ratio": cfg.task.fictitious_force_gravity_ratio
            }
            plot_evaluation_results(**asap_plot_args, traj_idx=traj_idx, save_dir=detailed_plot_dir, mode="ASAP")


        # Save trajectory data
        '''
        traj_data_dir = os.path.join(results_dir, "full_trajectories"); os.makedirs(traj_data_dir, exist_ok=True)
        np.savez(os.path.join(traj_data_dir, f"traj{traj_idx}_vanilla_data.npz"), **{k: v for k, v in vanilla_results.items() if k != 'errors'})
        np.savez(os.path.join(traj_data_dir, f"traj{traj_idx}_asap_data.npz"), **{k: v for k, v in asap_results.items() if k != 'errors'})
        '''

    # 6. Final Report
    # '''
    generate_summary_table(all_comparison_results, checkpoints_s, results_dir, dt)
    # '''

    simulation_app.close()
    print("Evaluation complete.")

if __name__ == "__main__":
    main()