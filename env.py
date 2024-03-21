import numpy as np
import rand_param_envs.gym as gym
from rand_param_envs.gym import spaces
import pandas as pd
from copy import deepcopy



class MetaLLM(gym.Env):
    def __init__(self, tasks: list=None, n_tasks: int = None, randomize_tasks=True):
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
        self.tasks = tasks
        self._task = self.tasks[0] # By default, the first task is selected
        self.n_tasks = len(self.tasks)
        self.set_task_idx(0)
        self._max_episode_steps = 3
        self._action_space = spaces.Discrete(26)  # 26 possible letters A-Z
        self.observation_space = spaces.MultiDiscrete([[1,26]] * 3) 
        self.goal_words = self._task['goal_words']
        self.whole_words = self._task['whole_words']
        self.word_length = self._task['word_length']
        self.current_word = ['#'] * 3  # Initialize empty word of length 3, '#' is a placeholder
        self.current_word_num = [-1] * 3  # Initialize empty word of length 3, '#' is a placeholder
        self.position = 0  # Initialize position of first empty letter
        self.sampling_kernels = self.estimate_sampling_kernels()
    
    def set_task(self, task):
        self._task = task
        self.goal_words = self._task['goal_words']
        self.whole_words = self._task['whole_words']
        self.word_length = self._task['word_length']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])
    
    def reset_task(self, idx):
        self.set_task(self.tasks[idx])

    def step(self, action):
        assert self.action_space.contains(action), f"{action} is an invalid action"

        # Convert action into corresponding letter
        letter = chr(97 + action)  # Convert action to letter (0 -> a, 1 -> b, ..., 25 -> z)

        self.current_word[self.position] = letter
        self.current_word_num[self.position] = action

        done = (self.position == 2)  # Done if all letters are filled (position is 2)
        reward = self.compute_reward()
        self.position += 1
        return deepcopy(self.current_word_num), reward, done, {'success': self.goal_words.__contains__(''.join(self.current_word))}

    def reset(self, seed=42):
        self.current_word = ['#'] * 3  # Reset word to empty
        self.position = 0 # Reset position to first letter
        self.current_word_num = [-1] * 3  # Reset word to empty
        self.seed(seed)
        return deepcopy(self.current_word_num)

    def render(self, mode='human'):
        print(''.join(self.current_word))
    
    def seed(self, seed=42):
        np.random.seed(seed)
        

    def render_rollouts(self, rollouts):
        pass

    def compute_reward(self):
        word_formed = ''.join(self.current_word)

        if (self.position == 2): # If the word is complete
            if word_formed in self.goal_words: # If the word is in the goal words
                return 1
            else:
                if not any(word.startswith(word_formed[:2]) for word in self.goal_words): # If the word is not in the goal words and no word starts with the current word
                    return -0.1
        else: # If the word is not complete
            if any(word.startswith(word_formed[:2]) for word in self.goal_words):
                return 0.1
        
        return 0
    
    def estimate_sampling_kernels(self):
        # Convert word list to a DataFrame
        alphabet_list = list('abcdefghijklmnopqrstuvwxyz')
        df = pd.DataFrame([list(word) for word in self.whole_words], columns=['First', 'Second', 'Third'])
        # Kernel 1: First letter probabilities
        kernel_1 = df['First'].value_counts(normalize=True).reindex(alphabet_list, fill_value=0).values
        kernels = [kernel_1]
        for col_index in range(self.word_length - 1):
            cross_tab = pd.crosstab(df.iloc[:, col_index], df.iloc[:, col_index + 1], normalize='index').reindex(index=alphabet_list, columns=alphabet_list, fill_value=0)
            # Some words may not contain certain letters, so we normalize the transition probabilities
            cross_tab = cross_tab.div(cross_tab.sum(axis=1), axis=0).fillna(0)
            kernels.append(cross_tab.values)
        return kernels

    def sample(self):
        if self.position == 0:
            return np.random.choice(26, p=self.sampling_kernels[0])
        else:
            return np.random.choice(26, p=self.sampling_kernels[self.position][ord(self.current_word[self.position - 1]) - 97])
    def dataset(self, num_episodes, discount_factor=0.99, seed=42):
        np.random.seed(seed)
        #columns: 'actions', 'discount_factor', 'mc_rewards', 'next_obs', 'obs', 'rewards', 'terminal_discounts', 'terminal_obs', 'terminals'
        data = {
            'actions': [],
            'discount_factor': discount_factor,
            'mc_rewards': [],
            'next_obs': [],
            'obs': [],
            'rewards': [],
            'terminal_discounts': [],
            'terminal_obs': [],
            'terminals': []
        }
        
        for episode in range(num_episodes):
            obs = deepcopy(self.reset())
            episode_data = []
            done = False
            while not done:
                action = self.sample()
                next_obs, reward, done, info = self.step(action)
                # print("obs", obs)
                # print("action", action)
                # print("reward", reward)
                # print("next_obs", next_obs)
                # print("done", done)
                episode_data.append((obs, action, reward, next_obs, done))
                obs = next_obs

            episode_rewards = [x[2] for x in episode_data]
            G = 0
            mc_rewards = [G:=G*discount_factor+reward for reward in episode_rewards[::-1]][::-1]
            terminal_obs = [episode_data[-1][3] for _ in range(len(episode_data))]
            terminal_discounts = [discount_factor**i for i in range(1, len(episode_data)+1)][::-1]

            data['actions'].extend([x[1] for x in episode_data])
            data['mc_rewards'].extend(mc_rewards)
            data['next_obs'].extend([x[3] for x in episode_data])
            data['obs'].extend([x[0] for x in episode_data])
            data['rewards'].extend([x[2] for x in episode_data])
            data['terminal_discounts'].extend(terminal_discounts)
            data['terminal_obs'].extend(terminal_obs)
            data['terminals'].extend([x[4] for x in episode_data])
        return data
    

    def get_all_task_idx(self):
        return range(len(self.tasks))

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, value):
        self._action_space = value
