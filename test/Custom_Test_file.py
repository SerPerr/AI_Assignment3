import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.MCTS import InformedMCTSAgent
from agents.dqn import DQNAgent
import cfgs.cfg_dqn as cfg

import gymnasium as gym
import ns_gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate
from tqdm import tqdm


if __name__ == '__main__':

    # --- 1. Περιβάλλον ---
    #domain = 'CartPole-v1'
    domain = 'MountainCar-v0'
    env = gym.make(domain, render_mode='human')
    name, version = domain.split("-")

    scheduler_1 = ContinuousScheduler()
    scheduler_2 = PeriodicScheduler(period=3)
    update_function1 = IncrementUpdate(scheduler_1, k=1)
    update_function2 = RandomWalk(scheduler_2)

    if name == "CartPole":
        tunable_params = {"masspole": update_function1, "gravity": update_function2}
    elif name == "MountainCar":
        tunable_params = {"gravity": update_function1, "force": update_function1}

    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=True)

    # --- 2. Mentor ---
    obs, info = ns_env.reset()
    planning_env = ns_env.get_planning_env()

    params = dict(cfg.agent)
    params['state_size'] = env.observation_space.shape[0]
    params['action_size'] = env.action_space.n
    
    if name == "CartPole":
        model_file = 'DDQN_episode_13271959.pth'
    elif name == "MountainCar":
        model_file = 'DDQN_episode_2118775.pth'

    base_dir = Path(__file__).resolve().parents[1]
    params['model_path'] = str(base_dir / 'agents' / 'DDQN_models' / domain / model_file)
    
    mentor = DQNAgent(**params)

    # --- 3. Agent ---
    agent = InformedMCTSAgent(
        env=planning_env,
        mentor_agent=mentor,
        noise_level=0.0,    # δοκίμασε και 0.1, 0.3
        num_simulations=50
    )

    # --- 4. Loop ---
    for episode in tqdm(range(10)):
        obs, info = ns_env.reset()
        planning_env = ns_env.get_planning_env()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.act(obs, planning_env)
            obs, reward, done, truncated, info = ns_env.step(action)
            planning_env = ns_env.get_planning_env()
            episode_reward += reward

        print(f"Episode {episode+1} | Reward: {episode_reward}")