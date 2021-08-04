import math

import gym
import numpy as np
import torch

import utils
from policies import QPolicy


class TabQPolicy(QPolicy):
    def __init__(self, env, buckets, actionsize, lr, gamma, model=None):
        """
        Inititalize the tabular q policy

        @param env: the gym environment
        @param buckets: specifies the discretization of the continuous state space for each dimension
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate for the model update
        @param gamma: discount factor
        @param model (optional): Stores the Q-value for each state-action
            model = np.zeros(self.buckets + (actionsize,))

        """
        super().__init__(len(buckets), actionsize, lr, gamma)
        self.env = env
        self.buckets = buckets
        self.actionsize = actionsize
        self.lr = lr
        self.gamma = gamma
        self.model = model
        if self.model is None:
            self.model = np.zeros(self.buckets + (actionsize,))
            #print(self.model) # prints [0,0,0]
        self.N = np.zeros(self.buckets + (actionsize,))

    def discretize(self, obs):
        """
        Discretizes the continuous input observation

        @param obs: continuous observation
        @return: discretized observation
        """
        upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def qvals(self, states):
        """
        Returns the q values for the states.

        @param state: the state

        @return qvals: the q values for the state for each action.
        """
        numstates = len(states)
        ret = np.zeros((numstates,3))

        for i in range(numstates):
            state = states[i]
            obs = self.discretize(state)
            #print(obs)
            for action in [0,1,2]:
                #print(action)
                ret[i][action] = self.model[obs][action]
                #print(ret[i][action])
        #print("one call")
        #print(state)
        #print(ret)
        return ret

    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        #print(self.model)
        #print("next")
        obs = self.discretize(state)
        max_action = 0
        nobs = self.discretize(next_state)
        orig = self.model[obs][action]
        #print(orig)
        for a in range(len(self.model[nobs])):
            if self.model[nobs][a] > self.model[nobs][max_action]:
                max_action = a

        #print(max_action)
        #print(state)
        #print(obs)
        # print(done)
        # if done is True:
        #     print(state[0])
        # if state[0] >= 0.5:
        #     reward = 1
        if done is True and next_state[0] >= 0.5:
            reward = 1
            target = reward
        else:
            target = reward + self.gamma * (self.model[nobs][max_action])
        self.model[obs][action] += 0.15*(target - self.model[obs][action])
        self.N[obs][action]+=1
        C = 0.01
        self.lr = min(self.lr,C/(C+self.N[obs][action]))
        #print(self.lr)
        #print(orig)
        return (orig - target)**2

    def save(self, outpath):
        """
        saves the model at the specified outpath
        """
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('MountainCar-v0')

    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n
    policy = TabQPolicy(env, buckets=(18, 14), actionsize=actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'models/tabular.npy')
