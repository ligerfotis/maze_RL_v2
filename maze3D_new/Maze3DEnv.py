import random
# Reward functions
from game import rewards
# Virtual environment
from maze3D_new.gameObjects import *
from maze3D_new.assets import *
from maze3D_new.utils import checkTerminal, convert_actions
from maze3D_new.config import layout_up_right, layout_down_right, layout_up_left
# RL modules
from plot_utils.plot_utils import get_config

# the game layouts
layouts = [layout_down_right, layout_up_left, layout_up_right]


class ActionSpace:
    def __init__(self):
        # self.actions = list(range(0, 14 + 1))
        # self.shape = 1
        self.actions = list(range(0, 3))
        self.shape = 2
        self.actions_number = len(self.actions)
        self.high = self.actions[-1]
        self.low = self.actions[0]

    def sample(self):
        # return [random.sample([0, 1, 2], 1), random.sample([0, 1, 2], 1)]
        return np.random.randint(self.low, self.high + 1, 2)


class Maze3D:
    """The environment wrapper for the Maze3D game"""

    def __init__(self, config=None, config_file=None):
        # get the configuration dictionary
        self.config = get_config(config_file) if config_file is not None else config

        # choose randomly one starting point for the ball
        current_layout = random.choice(layouts)
        # get discrete input
        self.discrete_input = self.config['game']['discrete_input']
        # check if playing with agent or with 2 humans
        self.rl = True if 'SAC' in self.config.keys() else False
        # create the game board
        self.board = GameBoard(current_layout, self.discrete_input, self.rl)
        # create the key dictionary
        self.keys = {pg.K_UP: 1, pg.K_DOWN: 2, pg.K_LEFT: 4, pg.K_RIGHT: 8}
        # create conversion key dictionary
        self.conversion_keys = {pg.K_UP: 0, pg.K_DOWN: 1, pg.K_LEFT: 2, pg.K_RIGHT: 3}
        # boolean that check if game has finished
        self.done = False
        # get the initial state of the board
        self.observation = self.get_state()  # must init board first
        # get the action space
        self.action_space = ActionSpace()
        # get the shape of the observation space
        self.observation_shape = (len(self.observation),)
        # the fps to run the game in
        self.fps = 60
        # retrieve the reward
        rewards.main(self.config)

    def step(self, action_agent, timed_out, goal, action_duration, duration_pause):
        """
        Performs the action of the agent to the environment for action_duration time.
        Simultaneously, receives input from the user via the keyboard arrows.
        :param action_agent: the action of the agent. make sure it is compatible
        :param timed_out: bool variable. true if game has been timed out
        :param goal: the goal of the game
        :param action_duration: the duration of the agent's action on the game
        :param duration_pause: the duration of the paused time
        :return: a transition [observation, reward, done, train_fps, duration_pause, action_list]
        """
        tmp_time = time.time()
        actions = [0, 0, 0, 0]
        action_list = []  # to store all the agent-human action pairs performed to the game.
        # perform agent's action for action_duration time
        while (time.time() - tmp_time) < action_duration and not self.done:
            # get keyboard action from user
            duration_pause, _, human_actions = self.getKeyboard(actions, duration_pause)
            action = [action_agent, human_actions[1]]
            action_list.append(action)

            self.board.handleKeys(action)  # apply action to the environment
            self.board.update()  # update board's rotations
            glClearDepth(1000.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.board.draw()  # render new graphics of the game
            pg.display.flip()

            clock.tick(self.fps)  # set the fps tick
            fps = clock.get_fps()  # get the actual fps performed

            pg.display.set_caption("Running at " + str(int(fps)) + " fps")
            self.observation = self.get_state()
            if checkTerminal(self.board.ball, goal) or timed_out:
                self.done = True
        reward = rewards.reward_function_maze(self.done, timed_out, ball=self.board.ball, goal=goal)
        return self.observation, reward, self.done, fps, duration_pause, action_list

    def getKeyboard(self, actions, duration_pause):
        """
        Retrieves human's action from keyboard arrows.
        -left/right
        -space: pause (press again to unpause)
        - up/down if applicable (in 2 humans set up)
        :param actions: an action vector used to convert actions
        :param duration_pause: the duration that the game was paused
        :return: duration_pause, action vector, human action
        """
        self.discrete_input = self.config['game']['discrete_input']
        if not self.discrete_input:
            pg.key.set_repeat(10)  # argument states the difference (in ms) between consecutive press events
        space_pressed = True
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return 1
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE and space_pressed:
                    space_pressed = False
                    # print("space")
                    start_pause = time.time()
                    pause()
                    end_pause = time.time()
                    duration_pause += end_pause - start_pause
                if event.key == pg.K_q:
                    exit(1)
                if event.key in self.keys:
                    actions[self.conversion_keys[event.key]] = 1
                    # action_human += maze.keys[event.key]
            if event.type == pg.KEYUP:
                if event.key in self.keys:
                    actions[self.conversion_keys[event.key]] = 0
                    # action_human -= maze.keys[event.key]
        human_actions = convert_actions(actions)
        return duration_pause, actions, human_actions

    def get_state(self):
        """
        ball pos x | ball pos y | ball vel x | ball vel y|  theta(x) | phi(y) |  theta_dot(x) | phi_dot(y)
        :return: the current state of the board
        """
        return np.asarray(
            [self.board.ball.x, self.board.ball.y, self.board.ball.velocity[0], self.board.ball.velocity[1],
             self.board.rot_x, self.board.rot_y, self.board.velocity[0], self.board.velocity[1]])

    def reset(self):
        """
        Resets the game
        :return: the initial observation of the game
        """
        self.__init__(config=self.config)
        return self.observation
