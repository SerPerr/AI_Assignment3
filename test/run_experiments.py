import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import ns_gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate
from agents.MCTS import MCTSUCTAgent, InformedMCTSAgent
from agents.dqn import DQNAgent
import cfgs.cfg_dqn as cfg
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# -------------------------------------------------------
# Βοηθητικές συναρτήσεις
# -------------------------------------------------------

def make_env(domain):
    """Δημιουργεί το ns περιβάλλον - render_mode=None για ταχύτητα"""
    env = gym.make(domain, render_mode=None)
    name, _ = domain.split("-")

    scheduler_1 = ContinuousScheduler()
    scheduler_2 = PeriodicScheduler(period=3)
    update_function1 = IncrementUpdate(scheduler_1, k=1)
    update_function2 = RandomWalk(scheduler_2)

    if name == "CartPole":
        tunable_params = {"masspole": update_function1, "gravity": update_function2}
    elif name == "MountainCar":
        tunable_params = {"gravity": update_function1, "force": update_function1}

    return NSClassicControlWrapper(env, tunable_params, change_notification=True)


def load_mentor(domain, ns_env):
    base_dir = Path(__file__).resolve().parents[1]

    if domain == "CartPole-v1":
        model_file = "DDQN_episode_13271959.pth"
    elif domain == "MountainCar-v0":
        model_file = "DDQN_episode_2118775.pth"

    model_path = str(base_dir / "agents" / "DDQN_models" / domain / model_file)

    params = dict(cfg.agent)
    params['state_size'] = ns_env.observation_space.shape[0]  # ← διόρθωση
    params['action_size'] = ns_env.action_space.n
    params['model_path'] = model_path

    return DQNAgent(**params)


def run_episode(agent, ns_env):
    """Τρέχει ένα επεισόδιο και επιστρέφει το cumulative reward"""
    obs, _ = ns_env.reset()
    done = False
    truncated = False
    total_reward = 0.0

    while not (done or truncated):
        planning_env = ns_env.get_planning_env()
        action = agent.act(obs, planning_env)
        obs, reward, done, truncated, _ = ns_env.step(action)
        total_reward += reward

    return total_reward


# -------------------------------------------------------
# Κύριο πείραμα
# -------------------------------------------------------

N_EPISODES   = 500
N_RUNS       = 3
NOISE_LEVELS = [0.0, 0.1, 0.3]
NUM_SIMS     = 10   # μειωμένο για ταχύτητα

domains = ["CartPole-v1", "MountainCar-v0"]

# Αποθηκεύουμε τα αποτελέσματα εδώ:
# results[domain][label] = λίστα με N_RUNS λίστες, κάθε μία με N_EPISODES rewards
results = {}

for domain in domains:
    print(f"\n=== Domain: {domain} ===")
    results[domain] = {}

    # Φορτώνουμε τον mentor μία φορά για το domain
    tmp_env = make_env(domain)
    mentor = load_mentor(domain, tmp_env)
    tmp_env.close()

    # ---- MCTS + UCT ----
    label = "MCTS+UCT"
    results[domain][label] = []

    for run in range(N_RUNS):
        print(f"  {label} | Run {run+1}/{N_RUNS}")
        ns_env = make_env(domain)
        run_rewards = []

        for ep in tqdm(range(N_EPISODES), desc=f"    Episodes"):
            planning_env = ns_env.get_planning_env()
            agent = MCTSUCTAgent(env=planning_env, num_simulations=NUM_SIMS)
            reward = run_episode(agent, ns_env)
            run_rewards.append(reward)

        ns_env.close()
        results[domain][label].append(run_rewards)

    # ---- MCTS + InformedUCT (για κάθε noise level) ----
    for noise in NOISE_LEVELS:
        label = f"InformedMCTS (noise={noise})"
        results[domain][label] = []

        for run in range(N_RUNS):
            print(f"  {label} | Run {run+1}/{N_RUNS}")
            ns_env = make_env(domain)
            run_rewards = []

            for ep in tqdm(range(N_EPISODES), desc=f"    Episodes"):
                planning_env = ns_env.get_planning_env()
                agent = InformedMCTSAgent(
                    env=planning_env,
                    mentor_agent=mentor,
                    noise_level=noise,
                    num_simulations=NUM_SIMS
                )
                reward = run_episode(agent, ns_env)
                run_rewards.append(reward)

            ns_env.close()
            results[domain][label].append(run_rewards)


# -------------------------------------------------------
# Γραφήματα - ένα ανά domain
# -------------------------------------------------------

for domain in domains:
    plt.figure(figsize=(12, 6))

    colors = ["blue", "orange", "green", "red"]

    for idx, (label, runs) in enumerate(results[domain].items()):
        runs_array = np.array(runs)          # shape: (N_RUNS, N_EPISODES)
        mean = runs_array.mean(axis=0)       # μέσος όρος across runs
        std  = runs_array.std(axis=0)        # τυπική απόκλιση across runs
        episodes = np.arange(1, N_EPISODES + 1)

        color = colors[idx % len(colors)]
        plt.plot(episodes, mean, label=label, color=color)
        plt.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)

    plt.xlabel("Επεισόδιο")
    plt.ylabel("Cumulative Reward")
    plt.title(f"{domain} - Σύγκριση Αλγορίθμων")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{domain.replace('-', '_')}_results.png", dpi=150)
    plt.show()
    print(f"Γράφημα αποθηκεύτηκε: {domain.replace('-', '_')}_results.png")