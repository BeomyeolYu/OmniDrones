import numpy as np
import os
from tqdm import tqdm
import glob

def clean_trajectory_data(source_dir, target_dir):
    """
    Loads all .npz chunks from a source directory. For each trajectory,
    it discards the first invalid timestep and saves the aligned, corrected
    trajectory to the target directory.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")

    chunk_files = glob.glob(os.path.join(source_dir, "*.npz"))
    if not chunk_files:
        print(f"No .npz files found in {source_dir}")
        return

    print(f"Found {len(chunk_files)} chunks to clean.")
    for filepath in tqdm(chunk_files, desc="Cleaning Data Chunks"):
        try:
            with np.load(filepath, allow_pickle=True) as data:
                # Load all original data arrays
                original_states = data['states']
                original_actions = data['actions']
                original_next_states = data['next_states']
                original_dones = data['dones']

                # Prepare lists for the new, cleaned trajectories
                cleaned_states, cleaned_actions, cleaned_next_states, cleaned_dones = [], [], [], []

                for i in range(len(original_states)):
                    # THE FIX: Discard the first invalid (s, a, s') tuple.
                    # We start everything from the second element (index 1).
                    if len(original_states[i]) > 1:
                        cleaned_states.append(original_states[i][1:])
                        cleaned_actions.append(original_actions[i][1:])
                        cleaned_next_states.append(original_next_states[i][1:])
                        cleaned_dones.append(original_dones[i][1:])

                # Save the cleaned data to the new directory
                new_filename = os.path.join(target_dir, "cleaned_" + os.path.basename(filepath))
                np.savez(
                    new_filename,
                    states=np.array(cleaned_states, dtype=object),
                    actions=np.array(cleaned_actions, dtype=object),
                    next_states=np.array(cleaned_next_states, dtype=object),
                    dones=np.array(cleaned_dones, dtype=object)
                )
        except Exception as e:
            print(f"Could not process {os.path.basename(filepath)}. Error: {e}")

    print("\nData cleaning complete.")
    print(f"Cleaned files are saved in the '{target_dir}' directory.")

if __name__ == "__main__":
    SOURCE_DIRECTORY = "deltaAction_trainingDataset"  # The folder with your NEW, misaligned data
    TARGET_DIRECTORY = "cleaned_delta_training_data" # A new folder for the fixed data
    clean_trajectory_data(SOURCE_DIRECTORY, TARGET_DIRECTORY)