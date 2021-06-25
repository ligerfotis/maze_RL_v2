## Human Input
The human input is given to the system through the keyboard arrows and can be either discrete or continuous. 
* Discrete input means that every keystroke produces one action. For another action, the human needs to release the button and press another one (or the same). It needs 5 keystrokes to reach the maximum rotation towards each direction from the starting pose of the tray.
* Continuous input means that a new human action is given as long as a key is pressed. For a pressed key, a new action will be available every ~15ms.

## Experiment
The main functions for running the game exist in the experiment.py
* `max_games_mode`: Runs the game with maximum number of episodes
* `max_interactions_mode`: Runs the game with maximum number of interactions
* `test_human`: Used in the case where the human plays alone
* `getKeyboard`: Parses the keyboard input
* `save_info`: Used for saving info e.g. the game reward, the game duration
* `get_action_pair`: Produces a vector with the human-agent joint action
* `save_experience`: Saves the experience, namely the states, actions and rewards that the agent witnesses in each episode
* `save_best_model`: Saves the best RL model
* `grad_updates`: Performs gradient updates in the training
* `print_logs`: Prints information for each episode
* `test_print_logs`: Prints information for the testing
* `compute_agent_action`: Computes the action of the agent
* `test_agent`: Used for testing the agent
* `get_agent_only_action`: Map the agent action to -1, 0 or 1. 
* `test_loop`: Not used


## Rewards
The following reward functions exist in the rewards.py:
* `reward_function_timeout_penalty`: For every non-terminal state the agent receives a reward of -1. In the goal state the agent receives a reward of 100. If the episode ends due to timeout, the agent gets -50.
* `reward_function_shafti`: For every non-terminal state the agent receives a reward of -1. In the goal state the agent receives a reward of 100.
* `reward_function_distance`: For every non-terminal state the agent receives a reward based on its distance from the goal. In the goal state the agent receives a reward of 100. If the episode ends due to timeout, the agent gets -50.
* `reward_function`: Template function in order to write a custom reward function.




