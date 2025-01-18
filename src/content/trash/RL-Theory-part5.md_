---
title: "Q-Learning and SARSA"
description: >-
  A detailed exploration of Q-Learning and SARSA methods for reinforcement
  learning, covering their working principles, differences, and applications.
pubDate: 2024-11-26
author: "Prof. Younes Jabrane"
category: "AI"
tags:
  - Q-Learning
  - SARSA
  - Reinforcement Learning
draft: true
---

# Q-Learning and SARSA

Q-Learning and SARSA are fundamental algorithms in reinforcement learning (RL). Both are designed to evaluate and improve an agent's policy but differ in their approaches to policy control.

---

## Q-Learning

### Overview
- **Type**: Off-policy control.
- **Objective**: Learn the optimal action-value function \( Q^*(s, a) \) by updating \( Q(s, a) \) values iteratively.
- **Characteristics**:
  - Model-free.
  - Uses a greedy policy for updates, irrespective of the agent's current policy.

### Update Rule
\[
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R(s_t, a_t) + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]
\]
- \( \alpha \): Learning rate.
- \( \gamma \): Discount factor.

### Process
1. **Initialization**:
   - Initialize \( Q(s, a) \) arbitrarily (commonly set to 0).
2. **Iteration**:
   - Start at a random state.
   - Select an action \( a_t \) (e.g., ε-greedy policy).
   - Observe the reward \( r_t \) and next state \( s_{t+1} \).
   - Update \( Q(s_t, a_t) \) using the update rule.

---

## SARSA

### Overview
- **Type**: On-policy control.
- **Objective**: Learn the action-value function \( Q(s, a) \) while following the agent's current policy.
- **Characteristics**:
  - Updates values using the agent's actual next action rather than the greedy action.

### Update Rule
\[
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R(s_t, a_t) + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
\]

### Process
1. **Initialization**:
   - Initialize \( Q(s, a) \) arbitrarily.
2. **Iteration**:
   - Start at a random state.
   - Select an action \( a_t \) using the current policy (e.g., ε-greedy).
   - Observe the reward \( r_t \) and next state \( s_{t+1} \).
   - Select the next action \( a_{t+1} \).
   - Update \( Q(s_t, a_t) \) using the update rule.

---

## Differences Between Q-Learning and SARSA

| Feature               | Q-Learning (Off-Policy)                            | SARSA (On-Policy)                            |
|-----------------------|---------------------------------------------------|---------------------------------------------|
| Policy Type           | Off-policy: Uses a greedy policy for updates.     | On-policy: Follows the current policy.       |
| Next Action           | Uses \( \max_a Q(s_{t+1}, a) \) for updates.       | Uses \( Q(s_{t+1}, a_{t+1}) \) for updates.  |
| Exploration-Exploitation Trade-off | Decoupled from updates.               | Coupled to updates through \( \epsilon \)-greedy. |
| Stability             | May converge faster but can be less stable.       | Converges more smoothly but slower.         |

---

## Expected SARSA (E-SARSA)

### Overview
- **Objective**: Combines SARSA with an expectation over all possible actions.
- **Update Rule**:
  \[
  Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R(s_t, a_t) + \gamma \sum_{a} \pi(a|s_{t+1}) Q(s_{t+1}, a) - Q(s_t, a_t) \right]
  \]
- **Key Difference**:
  - Computes the weighted average of \( Q(s_{t+1}, a) \) for all possible actions instead of using a single action \( a_{t+1} \).

---

## Application Example

### Environment
- **States (S)**: Rooms in a building.
- **Actions (A)**: Moves between rooms.
- **Rewards (R)**:
  - Positive reward for reaching a terminal state.
  - Negative reward for each step to encourage efficiency.

### Process
1. **Initialization**:
   - \( Q(s, a) = 0 \) for all \( s, a \).
2. **Iteration**:
   - Use the update rule for Q-Learning or SARSA to populate the \( Q \)-table.
3. **Policy Extraction**:
   - Extract the policy by choosing the action with the highest \( Q(s, a) \) for each state \( s \).

---

## Summary

| Method      | Type         | Update Rule                                                                 |
|-------------|--------------|-----------------------------------------------------------------------------|
| Q-Learning  | Off-Policy   | \( Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)] \) |
| SARSA       | On-Policy    | \( Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)] \) |
| E-SARSA     | On/Off-Policy| \( Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R + \gamma \sum_a \pi(a|s_{t+1}) Q(s_{t+1}, a) - Q(s_t, a_t)] \) |

Both Q-Learning and SARSA are effective RL methods, with their applicability depending on the problem's requirements for stability, exploration, and exploitation balance.