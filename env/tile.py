"""
Copied from https://github.com/horacepan/TileEnv/blob/master/tile_env/tile.py
"""

import time
import pandas as pd
import sys
import random
from gym import spaces
import gym
import numpy as np
import pdb

def neighbors(grid, x=None, y=None):
    '''
    grid: square numpy matrix
    x: x coord of the empty tile
    x: y coord of the empty tile
    Returns: list of neighbor states (grids)
    '''
    n = grid.shape[0]
    if x is None and y is None:
        empty_loc = np.where(grid == (n * n))
        x, y = empty_loc[0][0], empty_loc[1][0]

    nbrs = {}
    n = grid.shape[0]
    for m in TileEnv.MOVES:
        dx, dy = TileEnv.ACTION_MAP[m]
        new_x = x + dx
        new_y = y + dy
        if 0 <= new_x < n and 0 <= new_y < n:
            # continue
            grid[x][y], grid[new_x, new_y] = grid[new_x, new_y], grid[x][y]
            nbrs[m] = grid.copy()
            grid[x][y], grid[new_x, new_y] = grid[new_x, new_y], grid[x][y]
        else:
            nbrs[m] = grid.copy()

    return nbrs

def env_neighbors(env):
    return neighbors(env.grid, env.x, env.y)

def solveable(env):
    '''
    env: TileEnv
    A puzzle configuration is solveable if the sum of the permutation parity and the L1 distance of the
    empty tile to the corner location is even.

    If the grid width is odd, then the number of inversions in a solvable situation is even.
    If the grid width is even, and the blank is on an even row counting from the bottom (second-last, fourth-last etc), then the number of inversions in a solvable situation is odd.
    If the grid width is even, and the blank is on an odd row counting from the bottom (last, third-last, fifth-last etc) then the number of inversions in a solvable situation is even.

    Source:
    https://www.cs.bham.ac.uk/~mdr/teaching/modules04/java2/TilesSolvability.html
    '''
    perm = [i for i in env.perm_state() if i != (env.n * env.n)] # exclude empty tile
    if env.n % 2 == 1:
        return even_perm(perm)
    else:
        nth_from_bot = env.n - env._empty_x
        return ((n_inversions(perm) % 2 == 1) and (nth_from_bot % 2 == 0)) or \
               ((n_inversions(perm) % 2 == 0) and (nth_from_bot % 2 == 1))

def n_inversions(perm):
    '''
    perm: list/tuple of ints
    Returns: number of inversions of the given permutation
    '''
    n_invs = 0
    for idx, x in enumerate(perm):
        for ridx in range(idx+1, len(perm)):
            if x > perm[ridx]:
                n_invs += 1

    return n_invs

def even_perm(perm):
    '''
    perm: iterable
    Returns: True if perm is an even permutation
    '''
    return ((n_inversions(perm) % 2) == 0)

def random_perm(n):
    '''
    n: int, size of permutation
    Returns: A random permutation of A_n (an even permutation)
    '''
    x = list(range(1, n + 1))
    random.shuffle(x)
    return x

def grid_to_onehot(grid):
    '''
    Converts a n x n numpy matrix to a onehot representation of it.
    '''
    vec = np.zeros(grid.size * grid.size)
    idx = 0
    n = grid.shape[0]
    for i in range(n):
        for j in range(n):
            num = grid[i, j] - 1 # grid is 1-indexed
            vec[idx + num] = 1
            idx += grid.size

    return vec

class TileEnv(gym.Env):
    U = 0
    D = 1
    L = 2
    R = 3
    MOVES = [U, D, L, R]
    ACTION_MAP = {
        U: (-1, 0),
        D: (1, 0),
        L: (0, -1),
        R: (0, 1)
    }

    STR_ACTION_MAP = {
        U: 'U',
        D: 'D',
        L: 'L',
        R: 'R',
    }


    def __init__(self, n, one_hot=True, reward="sparse"):
        self.grid = np.array([i+1 for i in range(n * n)], dtype=int).reshape(n, n)
        self.n = n
        self.one_hot = one_hot
        self.action_space = spaces.Discrete(4)
        self.reward = reward

        if one_hot:
            self.observation_space = spaces.Box(low=0, high=1, shape=(n*n*n*n,))
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(n, n))

        self._empty_x = n - 1
        self._empty_y = n - 1
        self._valid_move_cache = self._init_valid_moves()
        self.onehot_shape = (n * n * n * n,)
        self.grid_shape = (n, n)

    def _init_valid_moves(self):
        valid = {}
        for i in range(self.n):
            for j in range(self.n):
                moves = []
                for m in TileEnv.MOVES:
                    dx, dy = TileEnv.ACTION_MAP[m]
                    new_x = i + dx
                    new_y = j + dy
                    if (0 <= new_x < self.n) and (0 <= new_y < self.n):
                        moves.append(m)

                valid[(i, j)] = moves
        return valid

    @property
    def actions(self):
        return self.action_space.n

    @property
    def x(self):
        return self._empty_x

    @property
    def y(self):
        return self._empty_y

    @staticmethod
    def solved_grid(n):
        contents = range(1, n*n + 1)
        grid = np.array(contents).reshape(n, n)
        return grid

    def _inbounds(self, x, y):
        return (0 <= x <= (self.n - 1)) and (0 <= y <= (self.n - 1))

    def get_reward(self, grid=None):
        if grid is None:
            grid = self.grid

        if self.reward == "sparse":
            return self.sparse_reward(grid)
        elif self.reward == 'penalty':
            return self.penalty_reward(grid)
        elif self.reward == 'penalty_sparse':
            return self.penalty_sparse_reward(grid)

    def step(self, action, ignore_oob=True):
        '''
        Actions: U/D/L/R
        ignore_oob: bool. If true, invalid moves on the boundary of the cube don't do anything.
        one_hot: bool.
            If true, this returns the one hot vector representation of the puzzle state
            If false, returns the grid representation.
        Move swap the empty tile with the tile in the given location
        '''
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        try:
            assert (self.grid[self._empty_x, self._empty_y] == (self.n * self.n))
        except:
            pdb.set_trace()

        dx, dy = TileEnv.ACTION_MAP[action]
        new_x = self._empty_x + dx
        new_y = self._empty_y + dy

        # what should we do with illegal moves?
        oob = not self._inbounds(new_x, new_y)
        if oob:
            if ignore_oob:
                #print('Taking step {} moves you oob! Not moving anything'.format(TileEnv.STR_ACTION_MAP[action]))
                done = self.is_solved()
                reward = self.get_reward()
                return self._get_state(), reward, done, {}
            else:
                raise Exception('Taking action {} will take you out of bounds'.format(action))

        # TODO: Make one hot state the default and construct grid only for rendering
        self.grid[new_x, new_y], self.grid[self._empty_x, self._empty_y] = self.grid[self._empty_x, self._empty_y], self.grid[new_x, new_y]
        self._empty_x = new_x
        self._empty_y = new_y
        done = self.is_solved()
        reward = self.get_reward()
        state = self._get_state()

        assert (self.grid[self._empty_x, self._empty_y] == (self.n * self.n))
        return state, reward, done, {}

    def penalty_reward(self, grid):
        return 1 if self.is_solved(grid) else -1

    def penalty_sparse_reward(self, grid):
        return 0 if self.is_solved(grid) else -1

    def sparse_reward(self, grid):
        return 1 if self.is_solved(grid) else 0

    def _pretty_print(self, x):
        if self.n <= 3:
            if x != (self.n * self.n):
                return '[{}]'.format(x)
            return '[_]'
        else:
            if x != (self.n * self.n):
                return '[{:2}]'.format(x)
            return '[__]'

    def render(self):
        for r in range(self.n):
            row = self.grid[r, :]
            for x in row:
                print(self._pretty_print(x), end='')
            print()

    def _get_state(self):
        if self.one_hot:
            return grid_to_onehot(self.grid)
        else:
            return self.grid

    def reset(self):
        '''
        Scramble the tile puzzle by taking some number of random moves
        This is actually really quite bad at scrambling
        '''
        self._assign_perm(random_perm(self.n * self.n))

        while not solveable(self):
            self._assign_perm(random_perm(self.n * self.n))

        assert (self.grid[self._empty_x, self._empty_y] == (self.n * self.n))
        return self._get_state()

    def shuffle(self, nsteps):
        '''
        nsteps: int
        Resets the state to a random state by taking {nsteps} random moves from the solved state
        '''
        ident_perm = tuple(i for i in range(1, self.n * self.n + 1))
        self._assign_perm(ident_perm)
        states = [(self.grid.copy(), self.x, self.y)] # always start at the solved state?

        for _ in range(nsteps):
            action = random.choice(self.valid_moves())
            state, _, _, _ = self.step(action)
            grid_state = self.grid.copy()
            states.append((grid_state, self.x, self.y))

        return states

    def _assign_perm(self, perm):
        '''
        perm: list or tuple representation of a S_{n*n} permutation.
        Assigns the grid state to the given permutation.
        This also properly sets the empty_x and empty_y positions.
        '''
        self.grid = np.array(perm, dtype=int).reshape(self.n, self.n)
        empty_loc = np.where(self.grid == (self.n * self.n))
        self._empty_x, self._empty_y = empty_loc[0][0], empty_loc[1][0]

    @staticmethod
    def from_perm(perm, one_hot=True):
        '''
        perm: tuple/list of ints
        Ex: The identity permutation for n = 4 is: (1, 2, 3, 4) and will yield the grid:
                [1][2]
                [3][4]
        '''
        n = int(np.sqrt(len(perm)))
        env = TileEnv(n, one_hot)
        env._assign_perm(perm)
        return env

    @staticmethod
    def static_is_solved(grid):
        n = grid.shape[0]
        idx = 1
        for i in range(n):
            for j in range(n):
                if grid[i, j] != idx:
                    return False
                idx += 1

        return True

    @staticmethod
    def is_solved_perm(tup):
        for i in range(1, len(tup) + 1):
            if tup[i] != i:
                return False
        return True

    def is_solved(self, grid=None):
        # 1-indexed
        if grid is None:
            grid = self.grid
        idx = 1
        n = grid.shape[0]
        for i in range(n):
            for j in range(n):
                if grid[i, j] != idx:
                    return False
                idx += 1

        return True

    def perm_state(self):
        return self.grid.ravel()

    def tup_state(self):
        return tuple(i for row in self.grid for i in row)

    @staticmethod
    def valid_move(action, grid, x=None, y=None):
        n = grid.shape[0]
        if x is None and y is None:
            empty_loc = np.where(grid == (n * n))
            x, y = empty_loc[0][0], empty_loc[1][0]

        dx, dy = TileEnv.ACTION_MAP[action]
        new_x = x + dx
        new_y = y + dy
        return (0 <= new_x < n) and (0 <= new_y < n)

    def _valid_move(self, action):
        return TileEnv.valid_move(action, self.grid, self.x, self.y)

    def valid_moves(self, x=None, y=None):
        #moves = [m for m in TileEnv.MOVES if self._valid_move(m)]
        if x is None:
            x = self.x
            y = self.y
        return self._valid_move_cache[(x, y)]

    # TODO: this is basically a copy of neighbors above. Consolidate
    def neighbors(self, grid=None, x=None, y=None):
        nbrs = {}
        if grid is None:
            grid = self.grid
            x, y = self.x, self.y
        if x is None:
            n = grid.shape[0]
            empty_loc = np.where(grid == (n * n))
            x, y = empty_loc[0][0], empty_loc[1][0]

        for m in self.valid_moves(x, y):
            dx, dy = TileEnv.ACTION_MAP[m]
            new_x = x + dx
            new_y = y + dy

            grid[x][y], grid[new_x, new_y] = grid[new_x, new_y], grid[x][y]
            nbrs[m] = grid.copy()
            grid[x][y], grid[new_x, new_y] = grid[new_x, new_y], grid[x][y]

        return nbrs

    # TODO: Should this be a static method?
    def peek(self, grid_state, x, y, action):
        n = grid_state.shape[0]
        dx, dy = TileEnv.ACTION_MAP[action]
        new_x = x + dx
        new_y = y + dy
        new_grid = grid_state.copy()
        if ((0 <= new_x < n) and (0 <= new_y < n)):
            new_grid[x][y], new_grid[new_x][new_y] = new_grid[new_x][new_y], new_grid[x][y]
        else:
            pass

        done = TileEnv.static_is_solved(new_grid)
        reward = -1 if not done else 1 # TODO: this is janky
        reward = self.get_reward(new_grid)
        return new_grid, reward, done, {'onehot': grid_to_onehot(new_grid)}

def grid_to_tup(grid):
    '''
    Get the permutation tuple representation of a grid
    Elements of grid are 1-indexed (contains values 1 - n**2), where n is the num rows of grid
    Returns tuple where index i of the tuple is where element i got mapped to

    Ex:
    (1, 2, 3, 4) is the permutation tuple corresponding to the grid:
    [1, 2]
    [3, 4]

    (3, 2, 4, 1) is the permutation corresponding to the grid:
    [4, 2]
    [1, 3]
    '''
    return tuple(i for row in grid for i in row)

def tup_to_onehot(tup):
    n = len(tup)
    onehot = np.zeros(n * n).astype(int)
    for i in range(n):
        val = tup[i] - 1
        onehot[i*n + val] = 1
    return onehot

def tup_to_grid(tup):
    n = int(np.sqrt(len(tup)))
    grid = np.zeros((n, n)).astype(int)
    idx = 0
    for i in range(n):
        for j in range(n):
            grid[i, j] = tup[idx]
            idx += 1
    return grid

def onehot_to_tup(onehot):
    n = int(np.sqrt(onehot.size))
    vals = []
    for idx, i in enumerate(np.where(onehot == 1)[0]):
        vals.append(i + 1 - (idx * n))
    return tuple(vals)

def test_peek():
    env = TileEnv(2)
    env._assign_perm([[1,2], [4,3]])
    ns, rew, done, _ = env.peek(env.grid, env.x, env.y, TileEnv.R)
    print(done)

if __name__ == '__main__':
    test_peek()
    n = 3
    env = TileEnv(n)
    env.shuffle(200)
    env.render()
