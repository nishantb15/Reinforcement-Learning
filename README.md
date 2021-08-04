# Reinforcement-Learning
 In the Mountain Car environment, a car is on a one-dimensional track, positioned between two mountains. The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum. The car's state is described by the observation space which is a 1D vector containing car's horizontal position and velocity. The car can take 3 actions: Left, Do Nothing, and Right. The episode ends when either the car reaches the flag on the right hill or it fails in 200 moves (or frames). All the move that fails to reach the flag will get reward -1. Therefore, in the episode where the car fails to reach to goal, the total reward would be -200. The entire episode will be scored as 1.0 if the car reaches the goal (total_reward>−200) or 0.0 otherwise.

If you don’t have the “gym” and “torch” packages, you may install them using pip:
pip3 install gym
pip3 install torch

Then run:
python3 tabular.py

To visualize the code:
python3 mp7.py –model models/tabular.npy

![image](https://github.com/nishantb15/Reinforcement-Learning/blob/main/Animated%20GIF-source.gif)
