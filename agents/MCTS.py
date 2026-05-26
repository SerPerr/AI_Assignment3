import gymnasium as gym
import numpy as np
from copy import deepcopy
import ns_gym as nsg

import ns_gym.base as base
import random


class RandomAgent(base.Agent):
    """A random agent that samples actions uniformly at random from the action space."""
    def __init__(self, env):
        self.env = env

    def act(self, observation, env):
        return self.env.action_space.sample()
    
class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.N = 0
        self.Q = 0.0
        self.prior = prior  
        
    def is_fully_expanded(self, num_actions):
        """
        Checks if we have tried all possible actions from this node.
        """
        return len(self.children) == num_actions
    
class MCTSUCTAgent(base.Agent):
    """Agent: Standard MCTS + UCT"""
    def __init__(self, env, num_simulations=100, c_constant=1.414, max_depth=200):
        self.env = env
        self.num_simulations = num_simulations
        self.c = c_constant
        self.max_depth = max_depth

    def act(self, observation, env):
        root = MCTSNode(state=observation)
        num_actions = env.action_space.n

        for _ in range(self.num_simulations):
            sim_env = deepcopy(env)
            node = root
            done = False
            truncated = False

            # --- 1. SELECTION ---
            while node.is_fully_expanded(num_actions) and len(node.children) > 0: # While we can keep going down the tree
                action = self._select_uct_child(node)               # Select the child with the highest UCT score
                obs, reward, done, truncated, info = sim_env.step(action) # Take the action in the simulated environment
                node = node.children[action]                        # Move down to the child node
                if done or truncated:                               # If we reach a terminal state during selection, we stop and will backpropagate the reward immediately
                    break

            # --- 2. EXPANSION ---
            if not done and not truncated: # If we are not at a terminal state, we can expand
                all_actions = set(range(num_actions)) # All possible actions from this node
                tried_actions = set(node.children.keys()) # Actions we have already tried from this node
                untried_actions = list(all_actions - tried_actions) # Actions we haven't tried from this node
                
                if len(untried_actions) > 0:  # If there are still untried actions, we expand by taking one of them
                    action = random.choice(untried_actions) # Randomly select one of the untried actions to expand
                    obs, reward, done, truncated, info = sim_env.step(action) # Take the action in the simulated environment
                    new_node = MCTSNode(state=obs, parent=node, action=action)  # Create a new node for the resulting state
                    node.children[action] = new_node    # Add the new node as a child of the current node
                    node = new_node                     # Move down to the new node

            # --- 3. SIMULATION / ROLLOUT ---
            total_reward = 0 # We will accumulate the reward obtained during the rollout
            depth = 0  
            while not (done or truncated) and depth < self.max_depth:    # While we haven't reached a terminal state and we haven't exceeded the maximum depth
                random_action = sim_env.action_space.sample() # Take a random action (rollout policy)
                obs, reward, done, truncated, info = sim_env.step(random_action) # Take the action in the simulated environment
                total_reward += reward  # Accumulate the reward obtained during the rollout
                depth += 1              # Increment the depth of the rollout

            # --- 4. BACKPROPAGATION ---
            while node is not None:     # Backpropagate the total reward obtained from the rollout up to the root
                node.N += 1             # Increment the visit count for this node
                node.Q += total_reward  # Add the total reward obtained from the rollout to the accumulated reward for this node
                node = node.parent      # Move up to the parent node


        def get_visits(action):
            return root.children[action].N # We want to select the action that has been visited the most (most simulations went through it) 

        best_action = max(root.children, key=get_visits)
        return best_action

    
    def _select_uct_child(self, node):
        """
        Selects the child node with the highest UCT score.
         UCT Score = (Q / N) + c * sqrt(log(N_parent) / N_child)
         where:
         - Q is the total reward accumulated in the child node
         - N is the number of times the child node has been visited
         - N_parent is the number of times the parent node has been visited
         - c is the exploration constant
        """
        best_score = -float('inf')
        best_action = None
        
        # for each child node, calculate the UCT score and keep track of the best one
        for action, child in node.children.items():
            
            # calculation of Exploitation (Average reward)
            exploitation = child.Q / child.N
            
            # calculation of Exploration (Based on visits)
            # We add a very small value (1e-6) to the denominator for safety
            # (even if the node is fully expanded, child.N is at least 1)
            exploration = self.c * np.sqrt(np.log(node.N) / (child.N + 1e-6))
            
            # Total UCT Score
            uct_score = exploitation + exploration
            
            # Keep the action that gives the highest score
            if uct_score > best_score:
                best_score = uct_score
                best_action = action
                
        return best_action



class InformedMCTSAgent(base.Agent):
    """Agent: Informed MCTS (AlphaGo Style)"""
    def __init__(self, env, mentor_agent, noise_level=0.0, num_simulations=100, c_constant=1.414, max_depth=200):
        self.env = env
        self.mentor_agent = mentor_agent
        self.noise_level = noise_level
        self.num_simulations = num_simulations
        self.c = c_constant
        self.max_depth = max_depth

    def act(self, observation, env):
        root = MCTSNode(state=observation)
        num_actions = env.action_space.n

        for _ in range(self.num_simulations):
            sim_env = deepcopy(env)
            node = root
            done = False
            truncated = False

            # --- 1. SELECTION ---
            # Διαβάζουμε το αποθηκευμένο prior από τον κόμβο - δεν καλούμε τον mentor
            while node.is_fully_expanded(num_actions) and len(node.children) > 0:
                action = self._select_puct_child(node)
                obs, reward, done, truncated, info = sim_env.step(action)
                node = node.children[action]
                if done or truncated:
                    break

            # --- 2. EXPANSION ---
            # Εδώ και μόνο εδώ καλούμε τον mentor για να υπολογίσουμε το prior
            if not done and not truncated:
                all_actions = set(range(num_actions))
                tried_actions = set(node.children.keys())
                untried_actions = list(all_actions - tried_actions)

                if len(untried_actions) > 0:
                    # Υπολογίζουμε το prior μία φορά για όλες τις ενέργειες
                    raw_state = node.state['state'] if isinstance(node.state, dict) else node.state
                    prior = self.mentor_agent.get_guidance(raw_state, noise_level=self.noise_level)

                    action = random.choice(untried_actions)
                    obs, reward, done, truncated, info = sim_env.step(action)
                    new_node = MCTSNode(
                        state=obs,
                        parent=node,
                        action=action,
                        prior=float(prior[action])  # ← αποθηκεύεται στον κόμβο
                    )
                    node.children[action] = new_node
                    node = new_node

            # --- 3. SIMULATION / ROLLOUT ---
            total_reward = 0
            depth = 0
            while not (done or truncated) and depth < self.max_depth:
                random_action = sim_env.action_space.sample()
                obs, reward, done, truncated, info = sim_env.step(random_action)
                total_reward += reward
                depth += 1

            # --- 4. BACKPROPAGATION ---
            while node is not None:
                node.N += 1
                node.Q += total_reward
                node = node.parent

        def get_visits(action):
            return root.children[action].N

        best_action = max(root.children, key=get_visits)
        return best_action

    def _select_puct_child(self, node):
        best_score = -float('inf')
        best_action = None

        for action, child in node.children.items():
            exploitation = child.Q / child.N
            exploration = self.c * child.prior * np.sqrt(node.N) / (1 + child.N)  # ← διαβάζουμε child.prior
            puct_score = exploitation + exploration

            if puct_score > best_score:
                best_score = puct_score
                best_action = action

        return best_action