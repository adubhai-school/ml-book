# Playing Atari with Deep Reinforcement Learning by Volodymyr Mnih et al.

Concepts required: Deep Learning, Gradient Descent, Convolutional Neural Networks, Reinforcement Learning, Q-Learning,  Experience Replay, Epsilon-Greedy Policy, Bellman Equation, RMSProp

Playing Atari with Deep Reinforcement Learning" is a paper published in 2013 by Volodymyr Mnih, et al. The paper presents a novel approach to developing artificial 
intelligence that is capable of playing Atari games at a superhuman level using deep reinforcement learning.

The researchers used a deep neural network to learn how to play Atari games by taking input from the game's screen pixels and outputting the appropriate action to 
take in the game. The deep neural network was trained using a combination of reinforcement learning and a technique called experience replay, which allowed the 
model to learn from past experiences.

The paper shows that their method, called the Deep Q-Network (DQN), can achieve human-level performance or better on several Atari games, including Space Invaders, 
Breakout, and Enduro, without any prior knowledge of the game mechanics or rules. Additionally, the DQN was able to learn to play the games directly from the raw 
pixel input, without requiring any feature engineering.

Overall, the paper demonstrates that deep reinforcement learning can be used to develop artificial intelligence systems capable of learning complex tasks directly 
from raw sensory input, without requiring explicit programming or knowledge of the underlying mechanics of the task.

## Key steps

1. Preprocessing: 

The raw pixel input from the Atari game is preprocessed into a format that the deep neural network can use as input. The preprocessing includes downsampling the image, 
converting it to grayscale, and stacking several frames of the game together to capture the motion of the game.

2. Deep neural network: 

A deep neural network is used to approximate the Q-function, which estimates the expected rewards for taking an action in a given state. The architecture used is a 
convolutional neural network (CNN), which is well-suited for processing images.

3. Reinforcement learning: 

The DQN uses a variant of Q-learning called deep Q-learning, which uses the deep neural network to approximate the Q-function. The DQN learns 
by taking actions in the game and receiving feedback in the form of rewards or penalties, which are used to update the neural network's weights.

4. Experience replay: 

The DQN uses a technique called experience replay to improve the stability and speed of learning. Experience replay involves storing 
transitions (state, action, reward, next state) in a replay buffer and randomly sampling from the buffer to train the neural network. This allows 
the network to learn from a diverse set of experiences and reduces the correlation between consecutive updates.

5. Training and evaluation: 

The DQN is trained on a variety of Atari games and evaluated based on its performance, measured in terms of the score achieved in the game. The DQN 
is able to learn to play the games at a superhuman level, achieving higher scores than human players on some games.

Overall, the process combines deep learning with reinforcement learning to enable the DQN to learn to play Atari games directly from raw pixel input. 
The DQN is able to achieve superhuman performance on several Atari games without any prior knowledge of the game mechanics or rules, demonstrating the 
potential of deep reinforcement learning for developing intelligent systems.

In the next sections we will create a model using pytorch and train it to play one atari game.
