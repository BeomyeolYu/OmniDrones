import pandas as pd
import matplotlib.pyplot as plt
import re

# Read the training log from file
with open("training_log.txt", "r") as file:
    log_data = file.read()

# Extract values using regex
pattern = r"Iteration (\d+): Avg Reward = ([\-\d\.]+), Avg Sim-Real Error = ([\d\.]+)"
matches = re.findall(pattern, log_data)

# Create DataFrame
df = pd.DataFrame(matches, columns=["Iteration", "Avg Reward", "Avg Sim-Real Error"])
df = df.astype({"Iteration": int, "Avg Reward": float, "Avg Sim-Real Error": float})

# Plot Iteration vs Avg Reward
plt.figure(figsize=(10, 5))
plt.plot(df["Iteration"], df["Avg Reward"], linewidth=3)
plt.title("Iteration vs Avg Reward")
plt.xlabel("Iteration (epoch; PPO T_horizon)")
plt.ylabel("Avg Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("iteration_vs_avg_reward.png")

# Plot Iteration vs Avg Sim-Real Error
plt.figure(figsize=(10, 5))
plt.plot(df["Iteration"], df["Avg Sim-Real Error"], linewidth=3)
plt.title("Iteration vs Avg Sim-Real Error")
plt.xlabel("Iteration (epoch; PPO T_horizon)")
plt.ylabel("Avg Sim-Real Error")
plt.grid(True)
plt.tight_layout()
plt.savefig("iteration_vs_sim_real_error.png")
