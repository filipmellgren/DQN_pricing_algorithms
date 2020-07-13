# DQN_pricing_algorithms
This repository contains code related to my master thesis "Tacit collusion with deep multi-agent reinforcement learning" written at the Stockholm School of Economics and the University of St. Gallen.

In short, the repository creates the output in the thesis by allowing two RL agents train against each other in a simulated environment. Note that training requires substantial time.

In practice, I had multiple versions running simultanouesly that had slightly different hyperparameters. I therefore exported the main folder with all the files, made the necessary change, and let it run. This repo contains one specific setup of hyperparameters. All changes happened in the "Main_notebook.ipynb" file (I think).

This repository became a mess towards the end of the thesis project as I made large changes quickly, did not know how to efficiently manage a change in approach I wasn't sure about (should probably have used branches).

Here is some indication of the intended structure (as I can remember it 4 months later) with the most important files:

* agent.py: contain an agent class that defines the notion of an agent, with its action.
* config.py: contains hyperparameters that are then imported elsewhere. It also contains utility functions such as a loss function, profit gain calculation function, and a normalisation of state function.
* experience_buffer.py: contains a class for an experience buffer that is held by the agents.
* cont_bertrand.py: implements the RL environment by describing transition dynamics. The name is supposed to refer to a "continuous bertrand" type of competition environment.
* dqn_model.py: contains a neural network setup that is used by the agents.
* calc_nash_monopoly.py: contains utility functions used to pin down Nash/monopoly prices/profits that are used to evaluate the agents. 
* training_seq.py: contains mega training loop and initialises the agents to train in the environment.
* testing_seq.py: is where two trained agents meet each other in a test scenario
* Main_notebook.ipynb: is a main file containing the highest level logic and calls on the other files to enable training and testing.
* test_agent.ipynb: was probably what I actually used to test agents.

Less important files include:
* actactprofit_mat.py: which creates a profit matrix for each pair of action.
* get_qs.py: extracts and exports Q-values


Note: the folder: Agent_v_static is basically irrelevant as it was used for a different approach.
