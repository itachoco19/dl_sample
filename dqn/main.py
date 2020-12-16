from collections import namedtuple
import random
import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
import gym
import animation

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

ENV = "CartPole-v0"
GANMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 500
BATCH_SIZE = 32
CAP = 10000

class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0
    
    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.memory[self.index] = Transition(state, action, state_next, reward)
            self.index = (self.index + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.num_states = num_states
        self.memory = ReplayMemory(CAP)
        self.model = self.create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0001)
    
    def Replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        self.model.eval()
        state_action_values = self.model(state_batch).gather(1, action_batch)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        next_state_values = torch.zeros(BATCH_SIZE)

        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + GANMA * next_state_values

        self.model.train()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
        return action
    
    def create_model(self):
        model = torch.nn.Sequential()
        model.add_module("fc1", torch.nn.Linear(self.num_states, 32))
        model.add_module("relu1", torch.nn.ReLU())
        model.add_module("fc2", torch.nn.Linear(32, 32))
        model.add_module("relu2", torch.nn.ReLU())
        model.add_module("fc3", torch.nn.Linear(32, self.num_actions))
        return model

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)
    
    def update_q_function(self):
        self.brain.Replay()
    
    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action
    
    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.agent = Agent(self.num_states, self.num_actions)
    
    def run(self):
        episode_10_list = np.zeros(10)
        complete_episodes = 0
        episode_final = False
        frames = []

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()
            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEPS):
                if episode_final is True:
                    frames.append(self.env.render(mode="rgb_array"))
                
                action = self.agent.get_action(state, episode)
                observation_next, _ , done, _ = self.env.step(action.item())

                if done:
                    state_next = None
                    
                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))
                    
                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes = complete_episodes + 1
                else:
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                self.agent.memorize(state, action, state_next, reward)

                self.agent.update_q_function()

                state = state_next

                if done:
                    print("{} Episode: Finished after {} steps : 10施行の平均ステップ数= {:.1f}".format(episode, step + 1, episode_10_list.mean()))
                    break
            if episode_final is True:
                animation.display_frames_as_gif(frames)
                break
            if complete_episodes >= 10:
                print("10回連続成功")
                episode_final = True

if __name__ == "__main__":
    cartpole_env = Environment()
    cartpole_env.run()