---
title: "Markov Decision Processes and Bellman Equations"
description: >-
  An in-depth exploration of Markov Decision Processes (MDPs) and the Bellman
  equations, covering their theoretical foundations and practical applications
  in Reinforcement Learning (RL).
pubDate: 2024-10-15
author: "Prof. Younes Jabrane"
category: Machine Learning
tags:
  - Markov Decision Process
  - Bellman Equations
  - Reinforcement Learning
---

# Markov Decision Processes and Bellman Equations

This chapter delves into the theoretical underpinnings of Reinforcement Learning, introducing key concepts such as Markov Chains (MC), Markov Reward Processes (MRP), and Markov Decision Processes (MDP). It also focuses on Bellman equations and their applications in evaluating agent behavior and improving decision-making strategies.

## Key Concepts

### Markov Chains (MC)
- A sequence of state transitions without considering rewards or actions.
- Defined by a transition matrix \( P \), where each entry \( P(s, s') \) represents the probability of transitioning from state \( s \) to \( s' \).

### Markov Reward Processes (MRP)
- Extends Markov Chains by incorporating rewards.
- Defined as \( (S, P, R, \gamma) \), where:
  - \( S \): Set of states.
  - \( P(s, s') \): Transition probabilities.
  - \( R(s, s') \): Reward for transitioning from \( s \) to \( s' \).
  - \( \gamma \): Discount factor (\( 0 \leq \gamma \leq 1 \)).

### Markov Decision Processes (MDP)
- Includes actions, enabling decision-making.
- Characterized by \( (S, A, P, R, \gamma) \):
  - \( A \): Set of actions.
  - \( P(s'|s, a) \): Probability of transitioning to \( s' \) from \( s \) after taking action \( a \).
  - \( R(s, a, s') \): Reward for taking action \( a \) in state \( s \) leading to \( s' \).

### Bellman Equations
- Central to RL, providing recursive relations for evaluating and optimizing policies.

#### State-Value Function (\( v_\pi(s) \)):
\[
v_\pi(s) = \mathbb{E}_\pi \left[ G_t | S_t = s \right] = \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma v_\pi(s') \right]
\]

#### Action-Value Function (\( q_\pi(s, a) \)):
\[
q_\pi(s, a) = \mathbb{E}_\pi \left[ R(s, a, s') + \gamma v_\pi(s') \right] = \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a'|s') q_\pi(s', a') \right]
\]

#### Optimality Equations:
- Optimal state-value function:
  \[
  v^*(s) = \max_a \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma v^*(s') \right]
  \]
- Optimal action-value function:
  \[
  q^*(s, a) = \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma \max_{a'} q^*(s', a') \right]
  \]

## Solving Bellman Equations

### Linear Programming
- Represented in matrix form:
  \[
  V_\pi = (I - \gamma P_\pi)^{-1} R_\pi
  \]

### Dynamic Programming
- Iteratively improves policies by alternating between policy evaluation and policy improvement:
  - **Policy Evaluation**: Computes \( v_\pi(s) \) for a fixed policy \( \pi \).
  - **Policy Improvement**: Updates the policy based on the action-value function \( q_\pi(s, a) \).

## Key Examples

### Rubik’s Cube
- A deterministic environment.
- The next state depends only on the current state and action, illustrating the Markov property.

### Atari Breakout
- A partially observable environment.
- Requires stacking multiple frames to deduce the ball's direction and restore the Markov property.

## Practical Considerations

### Model-Based vs. Model-Free
- **Model-Based**: Assumes knowledge of \( P \) and \( R \); uses dynamic programming.
- **Model-Free**: Learns \( P \) and \( R \) through interactions, suitable when the environment is unknown.

---

This chapter establishes the theoretical framework for MDPs and Bellman equations, laying the groundwork for advanced RL algorithms. Future discussions will address model-free approaches like Q-learning and policy gradients.