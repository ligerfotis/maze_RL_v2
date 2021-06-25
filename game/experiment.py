import csv
import math
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from statistics import mean

# Game utility file
from game.game_utils import get_env_action, get_distance_traveled, get_row_to_store, print_logs, test_print_logs, \
    column_names
# Offline Gradient Updates Scheduler
from game.updates_scheduler import UpdatesScheduler
# Virtual environment
from maze3D_new.assets import *

# to track memory leaks
from pympler.tracker import SummaryTracker
tracker = SummaryTracker()


class Experiment:
    def __init__(self, environment, agent=None, load_models=False, config=None):
        # retrieve parameters
        self.config = config  # configuration file dictionary
        self.env = environment  # environment to play in
        self.agent = agent  # the agent to play with
        self.df = pd.DataFrame(columns=column_names)
        if load_models:
            self.agent.load_models()

        # create a gradient updates scheduler
        self.scheduler = UpdatesScheduler()

        #retrieve information from the config file
        self.goal = config["game"]["goal"]
        self.mode = config['Experiment']['mode']
        self.action_duration = config['Experiment'][self.mode]['action_duration']
        self.max_game_duration = config['Experiment'][self.mode]['max_duration']
        self.max_games = config['Experiment'][self.mode]['max_games']
        self.log_interval = self.config['Experiment'][self.mode]['log_interval']
        self.isAgent_discrete = config['SAC']['discrete'] if 'SAC' in config.keys() else None
        self.second_human = config['game']['second_human'] if 'game' in config.keys() else None
        if not config['game']['human_alone']:
            self.max_score = config['Experiment']['test_loop']['max_score']
            self.test_model = config['game']['test_model']
            self.test_interval = config['Experiment']['test_interval']
            self.test_action_duration = config['Experiment']['test_loop']['action_duration']
            self.test_max_duration = config['Experiment']['test_loop']['max_duration']
            self.test_max_games = config['Experiment']['test_loop']['max_games']
            self.randomness_threshold = config['Experiment'][self.mode]['stop_random_agent']

        # fps tracking
        self.train_fps_list, self.test_fps_list, self.total_fps_list = [], [], []

        # initialize lists to keep track of information
        # train
        self.distance_travel_list, self.reward_list, self.game_duration_list = [], [], []
        self.online_update_duration_list, self.train_step_duration_list, self.grad_updates_durations = [], [], []
        self.action_history, self.score_history, self.episode_duration_list, self.length_list = [], [], [], []
        # test
        self.test_reward_list, self.test_step_duration_list, self.test_score_history = [], [], []
        self.test_episode_duration_list, self.test_length_list, self.test_distance_travel_list = [], [], []
        self.test_game_duration_list = []

        # initialize value variables
        self.best_score = -100 - 1 * self.max_game_duration
        self.best_reward = self.best_score
        self.best_score_episode, self.best_score_length = -1, -1
        self.human_actions, self.update_cycles = None, None
        self.save_models, self.flag = True, True
        self.test_game_number, self.train_game_number, self.duration_pause_total, self.total_steps = 0, 0, 0, 0

        # initialize train and test dataframes
        self.df_test = pd.DataFrame(columns=column_names)

    def max_games_mode(self):
        """
        In this experiment mode the user plays a number of games.
        Each game terminates either if  the goal is reached or if the maximum time duration of the game has passed.
        The flow is:
            0. (testing session with random agent) - if chosen
            1. training session, playing with random or pretrained agent
                1,5. Real-time Offline Gradient updates
            2. Offline gradient updates session
            3. testing session on recently trained agent
            4. Go To 1
        """
        train_step_counter = 0
        running_reward = 0
        avg_length = 0

        # 0. Perform a testing session with random agent
        if self.config['Experiment']['start_with_testing_random_agent']:
            self.test_max_games_mode(1)
            self.flag = True

        for i_game in range(1, self.max_games + 1):
            # print("Resuming Training")
            start_game_time = time.time()
            prev_observation, setting_up_duration = self.env.reset()  # stores the state of the environment
            timed_out = False  # used to check if we hit the maximum train_game_number duration
            game_reward = 0  # keeps track of the rewards for each train_game_number
            dist_travel = 0  # keeps track of the ball's travelled distance

            print("Episode: " + str(i_game))

            self.save_models = True  # flag for saving RL models
            redundant_end_duration = setting_up_duration  # duration in the game that is not playable by the user

            # 1. Training Session
            while True:
                train_game_start_time = time.time()
                self.total_steps += 1
                train_step_counter += 1  # keep track of the step number for each game

                env_agent_action, real_agent_action = self.get_agent_action(prev_observation, i_game)

                # Environment step
                transition = self.env.step(env_agent_action, timed_out, self.goal, self.action_duration)
                observation, reward, done, train_fps, duration_pause, action_list = transition

                redundant_end_duration += duration_pause  # keep track of the total paused time

                # check if the game has timed out
                if time.time() - start_game_time - redundant_end_duration >= self.max_game_duration:
                    timed_out = True

                # keep track of the fps
                self.train_fps_list.append(train_fps)
                self.total_fps_list.append(train_fps)

                self.action_history = self.action_history + action_list

                # add experience to buffer
                interaction = [prev_observation, real_agent_action, reward, observation, done]
                self.save_experience(interaction)

                game_reward += reward  # keep track of the total game reward

                # 1,5. Real-time Offline Gradient updates
                # online train (if applicable); checks performed in the function
                self.perform_online_grad_updates(i_game)

                # compute travelled distance
                dist_travel = get_distance_traveled(dist_travel, prev_observation, observation)

                # append row to the dataframe
                new_row = get_row_to_store(prev_observation)
                self.df = self.df.append(new_row, ignore_index=True)

                # calculate game duration
                train_step_duration = time.time() - train_game_start_time - duration_pause
                self.train_step_duration_list.append(train_step_duration)

                # set the observation for the next step
                prev_observation = observation

                # the ball has either reached the goal or the game has timed out
                if done:
                    break

            running_reward += game_reward  # total running reward. used for logging averages

            # keep track of best game reward
            self.update_best_reward(game_reward)

            # keep track of total pause duration
            end_game_time = time.time()
            self.update_time_metrics(start_game_time, end_game_time, redundant_end_duration)

            # update metrics about the experiment
            self.update_train_metrics(game_reward, dist_travel, train_step_counter)

            # 2. Offline gradient updates session
            self.offline_grad_updates_session(i_game)

            # 3. Testing Games Session
            self.testing_session(i_game)

            # logging
            avg_length += train_step_counter
            train_step_counter = 0
            avg_game_duration = np.mean(self.game_duration_list[-self.log_interval:])
            running_reward, avg_length = print_logs(self.config["game"]["verbose"],
                                                    self.config['game']['test_model'],
                                                    self.total_steps, i_game, self.total_steps, running_reward,
                                                    avg_length, self.log_interval, avg_game_duration)

        tracker.print_diff()  # to track memory leaks

    def test_max_games_mode(self, randomness_criterion):
        """
        In this testing experiment mode the user plays a number of games.
        Each game terminates either if  the goal is reached or if the maximum time duration of the game has passed.
        See max_interaction_mode description for more details
        :param randomness_criterion: the criterion to stop using random agent
        """
        test_step_counter = 0  # keep track of the step number for each game
        self.test_game_number += 1  # keep track of the testing session number
        print('Test {}'.format(self.test_game_number))
        best_score = 0  # best score during this session

        for game in range(1, self.test_max_games + 1):
            prev_observation, setting_up_duration = self.env.reset()  # get environment's initial state
            timed_out = False  # turn to false when the game has been timed out
            game_reward = 0  # the cumulative game reward
            dist_travel = 0  # the distance that the ball travels
            redundant_end_duration = setting_up_duration  # duration in the train_game_number that is not playable by the user
            start_game_time = time.time()  # the timestamp that the game starts
            while True:
                test_game_step_start_time = time.time()  # the step start time
                test_step_counter += 1  # keep track of the step number for each game

                env_agent_action, _ = self.get_agent_action(prev_observation, randomness_criterion)

                # Environment step
                transition = self.env.step(env_agent_action, timed_out, self.goal, self.test_action_duration)
                observation, _, done, test_fps, duration_pause, action_list = transition

                redundant_end_duration += duration_pause  # keep track of the total paused time

                if time.time() - start_game_time - redundant_end_duration >= self.test_max_duration:
                    timed_out = True

                # keep track of the fps
                self.test_fps_list.append(test_fps)
                self.total_fps_list.append(test_fps)

                # keep track of the action history. action_list contains every agent-human pair sent to the environment
                self.action_history = self.action_history + action_list

                # compute travelled distance
                dist_travel = get_distance_traveled(dist_travel, prev_observation, observation)

                # append row to the dataframe
                new_row = get_row_to_store(prev_observation)

                # append row to the dataframe
                self.df_test = self.df_test.append(new_row, ignore_index=True)

                # calculate game step duration
                test_step_duration = time.time() - test_game_step_start_time - duration_pause
                self.test_step_duration_list.append(test_step_duration)

                # subtract -1 for each step passed
                game_reward += -1

                # set the observation for the next step
                prev_observation = observation

                # the ball has either reached the goal or the game has timed out
                if done:
                    break

            # update the testing metrics
            best_score = self.update_test_metrics(redundant_end_duration, start_game_time, game_reward, dist_travel,
                                                  test_step_counter, best_score)

            test_step_counter = 0

        # todo: save best model
        # logging
        test_print_logs(mean(self.test_score_history[-10:]), mean(self.test_length_list[-10:]), best_score,
                        sum(self.test_episode_duration_list[-10:]))

    def test_human_max_games_mode(self):
        """
        In this experiment mode the user plays a number of games ALONE.
        Each game terminates either if  the goal is reached or if the maximum time duration of the game has passed.
        The flow is:
            1. training session
            2. testing session
            3. Go To 1
        """
        train_step_counter = 0
        running_reward = 0
        avg_length = 0

        for i_game in range(1, self.max_games + 1):
            # print("Resuming Training")
            start_game_time = time.time()
            prev_observation, setting_up_duration = self.env.reset()  # stores the state of the environment
            timed_out = False  # used to check if we hit the maximum train_game_number duration
            game_reward = 0  # keeps track of the rewards for each train_game_number
            dist_travel = 0  # keeps track of the ball's travelled distance

            print("Episode: " + str(i_game))

            self.save_models = True  # flag for saving RL models
            redundant_end_duration = setting_up_duration  # duration in the game that is not playable by the user

            # 1. Training Session
            while True:
                train_game_start_time = time.time()
                self.total_steps += 1
                train_step_counter += 1  # keep track of the step number for each game

                # Environment step
                transition = self.env.step(None, timed_out, self.goal, self.action_duration)
                observation, _, done, train_fps, duration_pause, action_list = transition
                reward = -1
                redundant_end_duration += duration_pause  # keep track of the total paused time

                # check if the game has timed out
                if time.time() - start_game_time - redundant_end_duration >= self.max_game_duration:
                    timed_out = True

                # keep track of the fps
                self.train_fps_list.append(train_fps)
                self.total_fps_list.append(train_fps)

                self.action_history = self.action_history + action_list

                game_reward += reward  # keep track of the total game reward

                # compute travelled distance
                dist_travel = get_distance_traveled(dist_travel, prev_observation, observation)

                # calculate game duration
                train_step_duration = time.time() - train_game_start_time - duration_pause
                self.train_step_duration_list.append(train_step_duration)

                # set the observation for the next step
                prev_observation = observation

                # the ball has either reached the goal or the game has timed out
                if done:
                    break

            running_reward += game_reward  # total running reward. used for logging averages

            # keep track of best game reward
            self.update_best_reward(game_reward)

            # keep track of total pause duration
            end_game_time = time.time()
            self.update_time_metrics(start_game_time, end_game_time, redundant_end_duration)

            # update metrics about the experiment
            self.update_train_metrics(game_reward, dist_travel, train_step_counter)

            # logging
            avg_length += train_step_counter
            train_step_counter = 0
            avg_game_duration = np.mean(self.game_duration_list[-self.log_interval:])
            running_reward, avg_length = print_logs(self.config["game"]["verbose"],
                                                    self.config['game']['test_model'],
                                                    self.total_steps, i_game, self.total_steps, running_reward,
                                                    avg_length, self.log_interval, avg_game_duration)

        tracker.print_diff()  # to track memory leaks

    def max_interactions_mode(self):
        pass

    def save_info(self, chkpt_dir, experiment_duration, total_games):
        """
        Saves experiment overall information in a file
        :param chkpt_dir: the checkpoint directory to store the file
        :param experiment_duration: the total duration of the experiment
        :param total_games: the total number of train games played
        """
        info = {'goal': self.goal, 'experiment_duration': experiment_duration, 'best_score': self.best_score,
                'best_score_episode': self.best_score_episode, 'best_reward': self.best_reward,
                'best_score_length': self.best_score_length, 'total_steps': self.total_steps,
                'total_games': total_games, 'fps': self.env.fps}
        w = csv.writer(open(chkpt_dir + '/rest_info.csv', "w"))
        for key, val in info.items():
            w.writerow([key, val])

    def save_experience(self, interaction):
        """
        Saves an interaction (prev_observation, agent_action, reward, observation, done) to the replay buffer of
        the agent.
        :param interaction: the interaction to be stored in the Replay Buffer
        """
        observation, agent_action, reward, observation_, done = interaction
        # we play with the RL agent
        if not self.second_human:
            if self.isAgent_discrete:
                self.agent.memory.add(observation, agent_action, reward, observation_, done)
            else:
                self.agent.remember(observation, agent_action, reward, observation_, done)

    def grad_updates(self, update_cycles):
        """
        Performs a number of offline gradient updates on the agent.
        :param update_cycles: the number of gradient updates to perform on the agent
        :return: the duration in sec of the gradient updates performed
        """
        start_grad_updates = time.time()
        end_grad_updates = 0
        # we play with the RL agent
        if not self.second_human:
            print("Performing {} updates".format(update_cycles))
            # print a completion bar in the terminal
            for _ in tqdm(range(update_cycles)):
                if self.isAgent_discrete:
                    # train the agent's networks
                    self.agent.learn()
                    # update the target networks
                    self.agent.soft_update_target()
                # continuous SAC agent
                else:
                    # train the agent's networks
                    self.agent.learn()
            end_grad_updates = time.time()

        return end_grad_updates - start_grad_updates

    def compute_agent_action(self, observation, randomness_criterion=None, randomness_threshold=None):
        """
        Computes agent's next action based on the observation given. It returns a random action from the legal ones
        until the randomness criterion has reach the randomness threshold.
        :param observation: the observation, based on which to calculate action
        :param randomness_criterion: the criterion we check to terminate random agent and start SAC agent
        :param randomness_threshold: threshold to stop random agent and start SAC agent
        :return: agent's action
        """
        if self.isAgent_discrete:
            # check if the criterion has reached the threshold to stop random exploration
            if randomness_criterion is not None and randomness_threshold is not None \
                    and randomness_criterion <= randomness_threshold:
                # Agent plays alone
                if self.config['game']['agent_only']:
                    agent_action = np.random.randint(pow(2, self.env.action_space.actions_number))
                # Agent plays with user
                else:
                    agent_action = np.random.randint(self.env.action_space.actions_number)
                self.save_models = False  # no point in saving the model since we use random exploration
                # flag is used to print only the first time that we start using random agent
                if self.flag:
                    print("Using Random Agent")
                    self.flag = False
            # use the agent to decide the next action
            else:
                self.save_models = True  # now it is time to start saving the model since we will be training it.
                agent_action = self.agent.actor.sample_act(observation)  # get agent's decision of the next action
                # flag is use to print message only the first time we start to use the agent
                if not self.flag:
                    print("Using SAC Agent")
                    self.flag = True
        # continuous SAC agent
        else:
            self.save_models = True
            agent_action = self.agent.choose_action(observation)  # get agent's decision of the next action

        return agent_action

    def update_best_reward(self, game_reward):
        """
        Updates the best reward so far
        :param game_reward: current game reward
        """
        if self.best_reward < game_reward:
            self.best_reward = game_reward

    def update_time_metrics(self, start_game_time, end_game_time, redundant_end_duration):
        """
        Updates experiment time duration metrics.
        """
        # the net total duration of the experiment
        self.duration_pause_total += redundant_end_duration
        game_duration = end_game_time - start_game_time - redundant_end_duration

        # keep track of the game duration
        self.game_duration_list.append(game_duration)

    def update_train_metrics(self, game_reward, dist_travel, current_timestep):
        """
        Updates train metrics
        :return:
        """
        # keep track of the game reward history
        self.reward_list.append(game_reward)

        # keep track of the ball's travelled distance
        self.distance_travel_list.append(dist_travel)

        # keep track of the game lentgh in steps
        self.length_list.append(current_timestep)

    def update_test_metrics(self, duration_pause, start_game_time, game_reward, dist_travel, step_counter, best_score):
        """
        Updates test metrics
        """
        end = time.time()

        self.duration_pause_total += duration_pause
        game_duration = end - start_game_time - duration_pause
        game_score = self.max_score + game_reward

        self.test_game_duration_list.append(game_duration)
        self.test_reward_list.append(game_reward)
        self.test_distance_travel_list.append(dist_travel)

        self.test_episode_duration_list.append(game_duration)
        self.test_score_history.append(game_score)
        self.test_length_list.append(step_counter)
        best_score = game_score if game_score > best_score else best_score
        return best_score

    def perform_online_grad_updates(self, i_game):
        """
        Performs real-time one offline gradient updates using a mini-batch from the replay buffer
        :param i_game: the current game. used to decide to use or not random agent
        """
        start_online_update = time.time()
        if not self.config['game']['test_model'] and not self.second_human:
            # check if we should do online updates
            if self.config['Experiment']['online_updates'] and i_game >= \
                    self.config['Experiment'][self.mode]['start_training_step_on_game']:
                if self.isAgent_discrete:
                    # train the agent's networks
                    self.agent.learn()
                    # update the target networks
                    self.agent.soft_update_target()
        # update the time needed for this update
        self.online_update_duration_list.append(time.time() - start_online_update)

    def ready_to_learn(self, i_game):
        """Checks if the agent is ready to learn, based on the current game"""
        return i_game >= self.config['Experiment'][self.mode]['start_training_step_on_game']

    def offline_grad_updates_session(self, i_game):
        """
        Perform a number of offline gradient updates.
        :param i_game: current game
        :return:
        """
        # if we do not test the model and the agent is ready to learn
        if not self.test_model and self.ready_to_learn(i_game):
            # check if it is time to learn
            if i_game % self.agent.update_interval == 0:
                print("off policy learning.")
                # get the number of cycles
                self.update_cycles = self.scheduler.schedule(self.max_game_duration, self.action_duration, self.mode,
                                                             self.max_games, self.update_cycles,
                                                             self.agent.update_interval, self.config)

                # print staff
                print("Update Cycles: {}".format(self.update_cycles))
                print("Max games: {}".format(self.max_games))
                print("Max duration of each game: {}".format(self.max_game_duration))

                if self.update_cycles > 0:
                    # perform an offline gradient update
                    grad_updates_duration = self.grad_updates(self.update_cycles)
                    # keep track of its duration
                    self.grad_updates_durations.append(grad_updates_duration)

                    # save the models after each grad update
                    self.agent.save_models()

    def testing_session(self, i_game):
        """
        Perform a number of test games.
        :param i_game: current game
        """
        if i_game % self.test_interval == 0 and self.test_max_games > 0:
            self.test_max_games_mode(randomness_criterion=None)
            print("Continue Training.")

    def get_agent_action(self, prev_observation, randomness_criterion):
        """
        Retrieves the original action from the agent and converts it into an environment-compatible one.
        Especially discrete SAC predicts actions using a categorical distribution, so we have to map these actions into
        both negative and positive numbers.
        :param prev_observation:
        :param randomness_criterion:
        :return: the environment_compatible agent's action, the original agent's action
        """
        # if playing with the agent and not with another human
        if not self.second_human:
            # compute agent's action
            agent_action = self.compute_agent_action(prev_observation, randomness_criterion,
                                                     self.randomness_threshold)

            # agent's action ready to be used from the environment
            env_agent_action = get_env_action(agent_action, self.isAgent_discrete)
        else:
            # get second human's action
            env_agent_action = None
            agent_action = None

        return env_agent_action, agent_action
