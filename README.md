# Deep-Reinforcement-Learning-Algorithms-for-Lunar-Lander

This repository contains the code and documentation for a comparative study on the performance of several deep reinforcement learning (DRL) algorithms in the Lunar Lander environment. The project sequentially implements and evaluates Deep Q-Network (DQN), Double DQN, Advantage Actor-Critic (A2C), and Soft Actor-Critic (SAC) algorithms.

## Table of Contents

- [Project Overview](#project-overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Future Work](#future-work)
- [Team Members](#team-members)


## Project Overview

This project investigates the effectiveness of different Deep Reinforcement Learning (DRL) algorithms in solving the Lunar Lander control problem. The goal is to land a simulated lunar lander safely on a designated landing pad. We sequentially implemented DQN, Double DQN, A2C, and SAC to observe how each algorithm improves upon its predecessors in terms of stability, learning speed, and final reward.

## Algorithms Implemented

- **Deep Q-Network (DQN):** A fundamental value-based RL algorithm using a neural network to approximate the Q-function.
- **Double Deep Q-Network (Double DQN):** An improvement over DQN, addressing the overestimation bias by using two Q-networks.
- **Advantage Actor-Critic (A2C):** An on-policy actor-critic algorithm combining policy-based and value-based methods.
- **Soft Actor-Critic (SAC):** A state-of-the-art off-policy algorithm that maximizes entropy along with reward to promote exploration and stability.

## Methodology

The project followed a systematic approach:

1.  **Sequential Implementation:** Algorithms were implemented sequentially, each building upon the previous one.
2.  **Hyperparameter Tuning:** Key hyperparameters were tuned based on prior work and experimentation.
3.  **Performance Evaluation:** The algorithms were evaluated on the basis of cumulative rewards, learning stability, and convergence speed.
4. **Comparative Analysis:** The results for all algorithms were compared and contrasted, using plots, numerical results and observations to identify best performing algorithms and patterns.

## Results

The following table summarizes the performance of the different algorithms in the Lunar Lander environment.

| Algorithm       | Learning Speed | Consistency | Final Avg. Reward (approx.) | Notes                                           |
|-----------------|--------------------|-------------|------------------------------|-------------------------------------------------|
| SAC             | Very Fast         | High         | >250                          | Best overall performance                        |
| DQN             | Slow               | Low          | >200                          | High variance, good final performance           |
| Double DQN      | Moderate          | Moderate     | >100 (possibly higher with more training) | Slightly better consistency than DQN            |
| A2C    | Slow               | Moderate     | ~0                             | Significantly underperforms other methods       |

For more detail on the results, please refer to the report.

## Key Findings

-   **SAC Achieves Highest Performance:** The Soft Actor-Critic (SAC) algorithm demonstrated the most robust and stable performance with the highest average cumulative reward.
-   **Double DQN Reduces Bias:** Double DQN showed more stable learning and better convergence compared to the vanilla DQN, which highlights the importance of addressing the overestimation bias.
-   **A2C Suffers from Instability:** The A2C method was less consistent and had more fluctuations during learning, demonstrating the inherent issue with on policy methods in these types of environments.
-   **Off-Policy methods do better**: Off-policy methods were better for the lunar lander environment than the A2C on-policy method.
-   **Exploration is Key:** Maximum entropy-based approaches (like SAC) promote better exploration, leading to better and more robust policies.


## Usage

1.  Clone the repository:
    ```bash
    git clone [repository link]
    ```
2.  Install the required dependencies (see below).
3.  Run the individual `ipynb` files.

## Dependencies

The project has the following dependencies. Make sure you have these installed or installed in a virtual environment:

-   Python 3.6+
-   PyTorch
-   Gymnasium (or OpenAI Gym)
-   Stable Baselines3 (for A2C)
-   NumPy
-   Matplotlib
-   Tensorboard (optional, for logging purposes)

Install with pip:

```bash
pip install torch gymnasium numpy matplotlib stable-baselines3 tensorboard
```

## Future Work

Future research directions may include:

*   Extensive hyperparameter tuning for all algorithms.
*   Deeper analysis of different exploration strategies.
*   Comparison with other state-of-the-art DRL algorithms.
*   Adaptation for more complex and real world environments.
*   Theoretical analysis of algorithm behaviour.

## Team Members

*   Saipraneeth Konuri
*   Samarth Sharma
*   Justin Cho


