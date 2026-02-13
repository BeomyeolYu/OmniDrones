import os
import matplotlib
# Use a backend that doesn't require a GUI, which is safer for servers.
# It will save the file but not block the script if a display isn't available.
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
"""
def plot_delta_action(trajectory_ar, trajectory_da, trajectory_ca, trajectory_s_r_next, trajectory_sim_next, save_path_prefix):
    trajectory_ar = np.array(trajectory_ar)
    trajectory_da = np.array(trajectory_da)
    trajectory_ca = np.array(trajectory_ca)
    trajectory_s_r_next = np.array(trajectory_s_r_next)
    trajectory_sim_next = np.array(trajectory_sim_next)

    time = np.arange(len(trajectory_ar))
    iteration_name = os.path.basename(save_path_prefix) # Extract the name for titles

    fig, axs = plt.subplots(4, 1, figsize=(12, 8))
    axs[0].plot(time, trajectory_ar[:, 0], 'r-', label='a_r[0]')
    axs[0].plot(time, trajectory_da[:, 0], 'b-', label='delta_a[0]')
    axs[0].plot(time, trajectory_ca[:, 0], 'g-', label='corrected_a[0]')
    axs[0].set_title(f'Action f over Time - Iteration {iteration_name}')
    axs[0].legend(); axs[0].grid()

    axs[1].plot(time, trajectory_ar[:, 1], 'r-', label='a_r[1]')
    axs[1].plot(time, trajectory_da[:, 1], 'b-', label='delta_a[1]')
    axs[1].plot(time, trajectory_ca[:, 1], 'g-', label='corrected_a[1]')
    axs[1].set_title(f'Action M_1 over Time - Iteration {iteration_name}')
    axs[1].legend(); axs[1].grid()

    axs[2].plot(time, trajectory_ar[:, 2], 'r-', label='a_r[2]')
    axs[2].plot(time, trajectory_da[:, 2], 'b-', label='delta_a[2]')
    axs[2].plot(time, trajectory_ca[:, 2], 'g-', label='corrected_a[2]')
    axs[2].set_title(f'Action M_2 over Time - Iteration {iteration_name}')
    axs[2].legend(); axs[2].grid()

    axs[3].plot(time, trajectory_ar[:, 3], 'r-', label='a_r[3]')
    axs[3].plot(time, trajectory_da[:, 3], 'b-', label='delta_a[3]')
    axs[3].plot(time, trajectory_ca[:, 3], 'g-', label='corrected_a[3]')
    axs[3].set_title(f'Action M_3 over Time - Iteration {iteration_name}')
    axs[3].legend(); axs[3].grid()

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_action.png")
    plt.close()

    # Figure 1: y and y_dot
    fig1, axs1 = plt.subplots(2, 3, figsize=(12, 6))
    for i in range(3):
        axs1[0, i].plot(time, trajectory_sim_next[:, i], label='sim y')
        axs1[0, i].plot(time, trajectory_s_r_next[:, i], '--', label='real y')

        axs1[0, i].set_title(f'next y{i+1} [m]')
        axs1[0, i].legend(); axs1[0, i].grid(True)

        axs1[1, i].plot(time, trajectory_sim_next[:, 3+i], label='sim y_dot')
        axs1[1, i].plot(time, trajectory_s_r_next[:, 3+i], '--', label='real y_dot')

        axs1[1, i].set_title(f'next y{i+1}_dot [m/s]')
        axs1[1, i].legend(); axs1[1, i].grid(True)
    fig1.tight_layout()
    fig1.savefig(f"{save_path_prefix}_next_y_ydot.png")
    plt.close(fig1)

    # Figure 2: q, w, W
    fig2, axs2 = plt.subplots(3, 3, figsize=(12, 8))
    for i in range(3):
        axs2[0, i].plot(time, trajectory_sim_next[:, 6+i], label='sim q')
        axs2[0, i].plot(time, trajectory_s_r_next[:, 6+i], '--', label='real q')

        axs2[0, i].set_title(f'next q{i+1}')
        axs2[0, i].legend(); axs2[0, i].grid(True)

        axs2[1, i].plot(time, trajectory_sim_next[:, 9+i], label='sim w')
        axs2[1, i].plot(time, trajectory_s_r_next[:, 9+i], '--', label='real w')

        axs2[1, i].set_title(f'next w{i+1}')
        axs2[1, i].legend(); axs2[1, i].grid(True)

        axs2[2, i].plot(time, trajectory_sim_next[:, 21+i], label='sim W')
        axs2[2, i].plot(time, trajectory_s_r_next[:, 21+i], '--', label='real W')

        axs2[2, i].set_title(f'next W{i+1}')
        axs2[2, i].legend(); axs2[2, i].grid(True)
    fig2.tight_layout()
    fig2.savefig(f"{save_path_prefix}_next_q_w_W.png")
    plt.close(fig2)

    # Figure 3: R matrix
    fig3, axs3 = plt.subplots(3, 3, figsize=(12, 8))
    R_labels = ['next R11', 'next R12', 'next R13', 'next R21', 'next R22', 'next R23', 'next R31', 'next R32', 'next R33']
    for i in range(3):
        for j in range(3):
            idx = 3 * i + j
            axs3[i, j].plot(time, trajectory_sim_next[:, 12 + idx], label='sim')
            axs3[i, j].plot(time, trajectory_s_r_next[:, 12 + idx], '--', label='real')
            axs3[i, j].set_title(R_labels[idx])
            axs3[i, j].legend(); axs3[i, j].grid(True)
    fig3.tight_layout()
    fig3.savefig(f"{save_path_prefix}_next_R.png")
    plt.close(fig3)

"""
def plot_delta_action(trajectory_ar, trajectory_da, trajectory_ca, trajectory_s_r_next, trajectory_sim_next, iteration):
    trajectory_ar = np.array(trajectory_ar)
    trajectory_da = np.array(trajectory_da)
    trajectory_ca = np.array(trajectory_ca)
    trajectory_s_r_next = np.array(trajectory_s_r_next)
    trajectory_sim_next = np.array(trajectory_sim_next)

    time = np.arange(len(trajectory_ar))

    fig, axs = plt.subplots(4, 1, figsize=(12, 8))
    axs[0].plot(time, trajectory_ar[:, 0], 'r-', label='a_r[0]')
    axs[0].plot(time, trajectory_da[:, 0], 'b-', label='delta_a[0]')
    axs[0].plot(time, trajectory_ca[:, 0], 'g-', label='corrected_a[0]')
    axs[0].set_title(f'Action f over Time - Iteration {iteration}')
    axs[0].legend(); axs[0].grid()

    axs[1].plot(time, trajectory_ar[:, 1], 'r-', label='a_r[1]')
    axs[1].plot(time, trajectory_da[:, 1], 'b-', label='delta_a[1]')
    axs[1].plot(time, trajectory_ca[:, 1], 'g-', label='corrected_a[1]')
    axs[1].set_title(f'Action M_1 over Time - Iteration {iteration}')
    axs[1].legend(); axs[1].grid()

    axs[2].plot(time, trajectory_ar[:, 2], 'r-', label='a_r[2]')
    axs[2].plot(time, trajectory_da[:, 2], 'b-', label='delta_a[2]')
    axs[2].plot(time, trajectory_ca[:, 2], 'g-', label='corrected_a[2]')
    axs[2].set_title(f'Action M_2 over Time - Iteration {iteration}')
    axs[2].legend(); axs[2].grid()

    axs[3].plot(time, trajectory_ar[:, 3], 'r-', label='a_r[3]')
    axs[3].plot(time, trajectory_da[:, 3], 'b-', label='delta_a[3]')
    axs[3].plot(time, trajectory_ca[:, 3], 'g-', label='corrected_a[3]')
    axs[3].set_title(f'Action M_3 over Time - Iteration {iteration}')
    axs[3].legend(); axs[3].grid()

    plt.tight_layout()
    plt.savefig(f"plot_DelAct_action_{iteration}.png")
    plt.close()

    # Figure 1: y and y_dot
    fig1, axs1 = plt.subplots(2, 3, figsize=(12, 6))
    for i in range(3):
        axs1[0, i].plot(time, trajectory_sim_next[:, i], label='sim y')
        axs1[0, i].plot(time, trajectory_s_r_next[:, i], '--', label='real y')

        axs1[0, i].set_title(f'next y{i+1} [m]')
        axs1[0, i].legend(); axs1[0, i].grid(True)

        axs1[1, i].plot(time, trajectory_sim_next[:, 3+i], label='sim y_dot')
        axs1[1, i].plot(time, trajectory_s_r_next[:, 3+i], '--', label='real y_dot')

        axs1[1, i].set_title(f'next y{i+1}_dot [m/s]')
        axs1[1, i].legend(); axs1[1, i].grid(True)
    fig1.tight_layout()
    fig1.savefig(f"plot_DelAct_next_y_ydot_{iteration}.png")
    plt.close(fig1)

    # Figure 2: q, w, W
    fig2, axs2 = plt.subplots(3, 3, figsize=(12, 8))
    for i in range(3):
        axs2[0, i].plot(time, trajectory_sim_next[:, 6+i], label='sim q')
        axs2[0, i].plot(time, trajectory_s_r_next[:, 6+i], '--', label='real q')

        axs2[0, i].set_title(f'next q{i+1}')
        axs2[0, i].legend(); axs2[0, i].grid(True)

        axs2[1, i].plot(time, trajectory_sim_next[:, 9+i], label='sim w')
        axs2[1, i].plot(time, trajectory_s_r_next[:, 9+i], '--', label='real w')

        axs2[1, i].set_title(f'next w{i+1}')
        axs2[1, i].legend(); axs2[1, i].grid(True)

        axs2[2, i].plot(time, trajectory_sim_next[:, 21+i], label='sim W')
        axs2[2, i].plot(time, trajectory_s_r_next[:, 21+i], '--', label='real W')

        axs2[2, i].set_title(f'next W{i+1}')
        axs2[2, i].legend(); axs2[2, i].grid(True)
    fig2.tight_layout()
    fig2.savefig(f"plot_DelAct_next_q_w_W_{iteration}.png")
    plt.close(fig2)

    # Figure 3: R matrix
    fig3, axs3 = plt.subplots(3, 3, figsize=(12, 8))
    R_labels = ['next R11', 'next R12', 'next R13', 'next R21', 'next R22', 'next R23', 'next R31', 'next R32', 'next R33']
    for i in range(3):
        for j in range(3):
            idx = 3 * i + j
            axs3[i, j].plot(time, trajectory_sim_next[:, 12 + idx], label='sim')
            axs3[i, j].plot(time, trajectory_s_r_next[:, 12 + idx], '--', label='real')
            axs3[i, j].set_title(R_labels[idx])
            axs3[i, j].legend(); axs3[i, j].grid(True)
    fig3.tight_layout()
    fig3.savefig(f"plot_DelAct_next_R_{iteration}.png")
    plt.close(fig3)

import numpy as np
import matplotlib.pyplot as plt

def plot_traj_collect(npz_file):
    """
    Plots trajectory data saved in a "jagged array" format (no padding).
    """
    print(f"Loading jagged data from {npz_file}...")
    # The 'allow_pickle=True' is essential for loading this format
    data = np.load(npz_file, allow_pickle=True)
    states = data["states"]
    next_states = data["next_states"]
    actions = data["actions"]
    
    # The data is now a list of arrays, not a single 3D array.
    NUM_EPISODES = len(states)
    print(f"Found {NUM_EPISODES} trajectories.")

    # --- 1. Setup All Figures ---
    fig1, axs1 = plt.subplots(4, 1, figsize=(15, 8))#, sharex=True)
    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 6))#, sharex=True)
    fig3, axs3 = plt.subplots(3, 3, figsize=(15, 8))#, sharex=True)
    fig4, axs4 = plt.subplots(3, 3, figsize=(15, 8))#, sharex=True)

    # --- 2. Main Plotting Loop ---
    # We iterate through each trajectory and plot it as a segment.
    time_offset = 0
    for i in range(NUM_EPISODES):
        # Get the data for this single trajectory
        traj_states = states[i]
        traj_next_states = next_states[i]
        traj_actions = actions[i]
        traj_len = len(traj_states)

        if traj_len == 0:
            continue

        # Create a time axis for this specific trajectory segment
        time = np.arange(time_offset, time_offset + traj_len)
        last_time_step = time[-1]

        # --- Plot Actions (Fig 1) ---
        for k in range(4):
            axs1[k].plot(time, traj_actions[:, k])
            axs1[k].scatter(last_time_step, traj_actions[-1, k], color='red', s=20, zorder=5)

        # --- Plot y and y_dot (Fig 2) ---
        for k in range(3):
            axs2[0, k].plot(time, traj_states[:, k], '--')
            axs2[0, k].plot(time, traj_next_states[:, k])
            axs2[0, k].scatter(last_time_step, traj_states[-1, k], color='red', s=20, zorder=5)
            
            axs2[1, k].plot(time, traj_states[:, 3 + k], '--')
            axs2[1, k].plot(time, traj_next_states[:, 3 + k])
            axs2[1, k].scatter(last_time_step, traj_states[-1, 3 + k], color='red', s=20, zorder=5)

        # --- Plot q, w, W (Fig 3) ---
        for k in range(3):
            axs3[0, k].plot(time, traj_states[:, 6 + k], '--')
            axs3[0, k].plot(time, traj_next_states[:, 6 + k])
            axs3[0, k].scatter(last_time_step, traj_states[-1, 6 + k], color='red', s=20, zorder=5)

            axs3[1, k].plot(time, traj_states[:, 9 + k], '--')
            axs3[1, k].plot(time, traj_next_states[:, 9 + k])
            axs3[1, k].scatter(last_time_step, traj_states[-1, 9 + k], color='red', s=20, zorder=5)

            axs3[2, k].plot(time, traj_states[:, 21 + k], '--')
            axs3[2, k].plot(time, traj_next_states[:, 21 + k])
            axs3[2, k].scatter(last_time_step, traj_states[-1, 21 + k], color='red', s=20, zorder=5)
        
        # --- Plot R matrix (Fig 4) ---
        for row in range(3):
            for col in range(3):
                idx = 12 + 3 * row + col
                axs4[row, col].plot(time, traj_states[:, idx], '--')
                axs4[row, col].plot(time, traj_next_states[:, idx])
                axs4[row, col].scatter(last_time_step, traj_states[-1, idx], color='red', s=20, zorder=5)

        # Update the time offset for the start of the next trajectory
        time_offset += traj_len

    # --- 3. Final Figure Formatting ---
    print("Formatting figures...")
    
    # Figure 1: Actions
    action_labels = ["f", "M1", "M2", "M3"]
    for i in range(4):
        axs1[i].set_ylabel(action_labels[i]); axs1[i].grid(True)
    fig1.suptitle("Concatenated Trajectories - Actions")
    fig1.tight_layout()
    fig1.savefig("plot_TragCollect_actions.png")

    # Figure 2: y and y_dot
    y_labels = ["y1", "y2", "y3"]
    for i in range(3):
        axs2[0, i].set_title(y_labels[i]); axs2[0, i].grid(True)
        axs2[1, i].set_title(y_labels[i] + "_dot"); axs2[1, i].grid(True)
    axs2[0,0].plot([], [], 'C0--', label='state') # Dummy plots for unified legend
    axs2[0,0].plot([], [], 'C1-', label='next_state')
    fig2.legend()
    fig2.tight_layout()
    fig2.savefig("plot_TragCollect_y_ydot.png")

    # Figure 3: q, w, W
    for i in range(3):
        axs3[0, i].set_title(f"q{i+1}"); axs3[0, i].grid(True)
        axs3[1, i].set_title(f"w{i+1}"); axs3[1, i].grid(True)
        axs3[2, i].set_title(f"W{i+1}"); axs3[2, i].grid(True)
    axs3[0,0].plot([], [], 'C0--', label='state')
    axs3[0,0].plot([], [], 'C1-', label='next_state')
    fig3.legend()
    fig3.tight_layout()
    fig3.savefig("plot_TragCollect_q_w_W.png")

    # Figure 4: R matrix
    R_labels = ['R11','R12','R13','R21','R22','R23','R31','R32','R33']
    for i in range(3):
        for j in range(3):
            axs4[i,j].set_title(R_labels[3*i+j]); axs4[i,j].grid(True)
    axs4[0,0].plot([], [], 'C0--', label='state')
    axs4[0,0].plot([], [], 'C1-', label='next_state')
    fig4.legend()
    fig4.tight_layout()
    fig4.savefig("plot_TragCollect_R.png")

    # --- 4. Cleanup ---
    plt.close('all')
    print("Saved 4 concatenated trajectory figures.")


'''
def plot_traj_collect(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    states = data["states"]
    next_states = data["next_states"]
    actions = data["actions"]
    dones = data["dones"]

    NUM_EPISODES, MAX_STEPS, _ = states.shape
    total_steps = NUM_EPISODES * MAX_STEPS
    time = np.arange(total_steps)
    print("len(time):", len(time))

    # Concatenate along episodes
    all_states = states.reshape(total_steps, -1).astype(float)
    all_next_states = next_states.reshape(total_steps, -1).astype(float)
    all_actions = actions.reshape(total_steps, -1).astype(float)
    all_dones = dones.reshape(total_steps).astype(bool)
    done_indices = np.where(all_dones)[0]

    # ---- BREAK LINES AT EPISODE BOUNDARIES ----
    adjusted_done_indices = []
    offset = 0
    for idx in done_indices:
        insert_at = idx + 1 + offset
        if insert_at < len(time) + offset:
            all_states = np.insert(all_states, insert_at, np.nan, axis=0)
            all_next_states = np.insert(all_next_states, insert_at, np.nan, axis=0)
            all_actions = np.insert(all_actions, insert_at, np.nan, axis=0)
            time = np.insert(time, insert_at, time[idx])
            adjusted_done_indices.append(idx + offset)  # mark the correct done location
            offset += 1  # because we inserted one extra row
    done_indices = np.array(adjusted_done_indices)

    # --------------------- Figure 1: Actions ---------------------
    fig1, axs1 = plt.subplots(4, 1, figsize=(12, 8))
    action_labels = ["f", "M1", "M2", "M3"]
    for i in range(4):
        axs1[i].plot(time, all_actions[:, i], label="action")
        axs1[i].scatter(time[done_indices], all_actions[done_indices, i], color='red', s=20)
        axs1[i].set_ylabel(action_labels[i])
        axs1[i].grid(True)
    fig1.suptitle("Concatenated Trajectories - Actions")
    fig1.tight_layout()
    fig1.savefig("plot_TragCollect_actions.png")
    plt.close(fig1)

    # --------------------- Figure 2: y and y_dot ---------------------
    fig2, axs2 = plt.subplots(2, 3, figsize=(12, 6))
    y_labels = ["y1", "y2", "y3"]
    for i in range(3):
        axs2[0, i].plot(time, all_states[:, i], '--', label="state")
        axs2[0, i].plot(time, all_next_states[:, i], label="next_state")
        axs2[0, i].scatter(time[done_indices], all_states[done_indices, i], color='red', s=20)
        axs2[0, i].set_title(y_labels[i])
        axs2[0, i].grid(True)

        axs2[1, i].plot(time, all_states[:, 3 + i], '--', label="state")
        axs2[1, i].plot(time, all_next_states[:, 3 + i], label="next_state")
        axs2[1, i].scatter(time[done_indices], all_states[done_indices, 3 + i], color='red', s=20)
        axs2[1, i].set_title(y_labels[i] + "_dot")
        axs2[1, i].grid(True)

    for ax in axs2.flatten():
        ax.legend()
    fig2.tight_layout()
    fig2.savefig("plot_TragCollect_y_ydot.png")
    plt.close(fig2)

    # --------------------- Figure 3: q, w, W ---------------------
    fig3, axs3 = plt.subplots(3, 3, figsize=(12, 8))
    for i in range(3):
        axs3[0, i].plot(time, all_states[:, 6 + i], '--', label="state")
        axs3[0, i].plot(time, all_next_states[:, 6 + i], label="next_state")
        axs3[0, i].scatter(time[done_indices], all_states[done_indices, 6 + i], color='red', s=20)
        axs3[0, i].set_title(f"q{i+1}"); axs3[0, i].grid(True)

        axs3[1, i].plot(time, all_states[:, 9 + i], '--', label="state")
        axs3[1, i].plot(time, all_next_states[:, 9 + i], label="next_state")
        axs3[1, i].scatter(time[done_indices], all_states[done_indices, 9 + i], color='red', s=20)
        axs3[1, i].set_title(f"w{i+1}"); axs3[1, i].grid(True)

        axs3[2, i].plot(time, all_states[:, 21 + i], '--', label="state")
        axs3[2, i].plot(time, all_next_states[:, 21 + i], label="next_state")
        axs3[2, i].scatter(time[done_indices], all_states[done_indices, 21 + i], color='red', s=20)
        axs3[2, i].set_title(f"W{i+1}"); axs3[2, i].grid(True)

    for ax in axs3.flatten():
        ax.legend()
    fig3.tight_layout()
    fig3.savefig("plot_TragCollect_q_w_W.png")
    plt.close(fig3)

    # --------------------- Figure 4: R matrix ---------------------
    fig4, axs4 = plt.subplots(3, 3, figsize=(12, 8))
    R_labels = ['R11','R12','R13','R21','R22','R23','R31','R32','R33']
    for i in range(3):
        for j in range(3):
            idx = 12 + 3*i + j
            axs4[i,j].plot(time, all_states[:, idx], '--', label="state")
            axs4[i,j].plot(time, all_next_states[:, idx], label="next_state")
            axs4[i,j].scatter(time[done_indices], all_states[done_indices, idx], color='red', s=20)
            axs4[i,j].set_title(R_labels[3*i+j]); axs4[i,j].grid(True)

    for ax in axs4.flatten():
        ax.legend()
    fig4.tight_layout()
    fig4.savefig("plot_TragCollect_R.png")
    plt.close(fig4)

    print("Saved 4 concatenated trajectory figures")
'''