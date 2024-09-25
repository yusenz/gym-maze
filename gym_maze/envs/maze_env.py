import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_maze.envs.maze_view_2d import MazeView2D


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True, screen_size=(320,320), penalty=0.1, penalty_normalize='size'):

        self.viewer = None
        self.enable_render = enable_render
        self.penalty = penalty
        self.penalty_normalize = penalty_normalize

        if maze_file:
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=screen_size, 
                                        enable_render=enable_render)
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size)/3))
            else:
                has_loops = False
                num_portals = 0

            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=screen_size,
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high =  np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if isinstance(action, int):
            self.maze_view.move_robot(self.ACTION[action])
        else:
            self.maze_view.move_robot(action)

        if np.array_equal(self.maze_view.robot, self.maze_view.goal):
            reward = 1
            terminated = True
            truncated = False
        else:
            # reward = -0.1/(self.maze_size[0]*self.maze_size[1])
            reward = -self.penalty
            if self.penalty_normalize == 'size':
                reward /= self.maze_size[0]*self.maze_size[1]
            elif self.penalty_normalize == 'sqrt_size':
                reward /= np.sqrt(self.maze_size[0]*self.maze_size[1])
            elif self.penalty_normalize == 'log_size':
                reward /= np.log(self.maze_size[0]*self.maze_size[1])
            elif self.penalty_normalize == 'none':
                pass
            else:
                raise ValueError("penalty_normalize must be one of 'size', 'sqrt_size', 'log_size', or 'none'")
            terminated = False
            # truncation is handled by gym.wrappers.TimeLimit
            truncated = False

        self.state = self.maze_view.robot

        info = {}

        return self.state, reward, terminated, truncated, info

    def reset(self):
        self.maze_view.reset_robot()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)


class MazeEnvSample5x5(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy", **kwargs)


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5), **kwargs)


class MazeEnvSample10x10(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy", **kwargs)


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10), **kwargs)


class MazeEnvSample3x3(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy", **kwargs)


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3), **kwargs)


class MazeEnvSample100x100(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy", **kwargs)


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100), **kwargs)


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus", **kwargs)


class MazeEnvRandom20x20Plus(MazeEnv):

    def __init__(self, **kwargs):
        super(MazeEnvRandom20x20Plus, self).__init__(maze_size=(20, 20), mode="plus", **kwargs)


class MazeEnvRandom30x30Plus(MazeEnv):
    def __init__(self, **kwargs):
        super(MazeEnvRandom30x30Plus, self).__init__(maze_size=(30, 30), mode="plus", **kwargs)
