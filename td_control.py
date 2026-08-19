"""
SARSA and Q-Learning on the same 4x4 gridworld used in 418_Code_1.py.

418_Code_1.py solves the gridworld with Dynamic Programming: it has a full
model of the environment (it knows actionRewardFunction exactly), so it can
sweep every one of the 16 states on every iteration and update them directly
from the Bellman equation.

This script instead treats the environment as a black box. The agent starts
at a random state and takes real actions, one at a time, only ever seeing
the (state, action, reward, next_state) it just experienced. It has to learn
the value function and policy purely from those sampled transitions -
this is Temporal-Difference (TD) control.

SARSA is on-policy: it updates Q(s,a) toward the value of the action it
will actually take next (also chosen epsilon-greedily).

Q-learning is off-policy: it updates Q(s,a) toward the value of the BEST
possible next action, even though the agent might not actually take that
action next (because of epsilon-greedy exploration).
"""

import random
import numpy as np

from viz_common import (
    SARSA_COLOR, QLEARN_COLOR, INK_PRIMARY, INK_SECONDARY,
    rollingMean, plotSeries, plotValueGrids,
)

# ---- same gridworld definition as 418_Code_1.py ----
gridSize = 4
terminationStates = [[0, 0], [gridSize - 1, gridSize - 1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]  # up, down, right, left
rewardSize = -1
gamma = 1.0


def step(state, actionIdx):
    """Apply actions[actionIdx] to state; return (nextState, reward)."""
    action = actions[actionIdx]
    nextState = (np.array(state) + np.array(action)).tolist()

    if -1 in nextState or gridSize in nextState:
        nextState = state  # move off the grid: bounce back to the same state

    return nextState, rewardSize


def randomStartState():
    """Sample a random non-terminal state to begin an episode from."""
    while True:
        state = [random.randint(0, gridSize - 1), random.randint(0, gridSize - 1)]
        if state not in terminationStates:
            return state


def epsilonGreedy(Q, state, epsilon):
    """With probability epsilon act randomly, otherwise act greedily on Q."""
    if random.random() < epsilon:
        return random.randrange(len(actions))
    return int(np.argmax(Q[state[0], state[1]]))


def runSARSA(numEpisodes=500, alpha=0.1, epsilon=0.1):
    Q = np.zeros((gridSize, gridSize, len(actions)))
    valueDeltas = []

    for _ in range(numEpisodes):
        prevV = np.max(Q, axis=2).copy()

        state = randomStartState()
        actionIdx = epsilonGreedy(Q, state, epsilon)

        while state not in terminationStates:
            nextState, reward = step(state, actionIdx)
            nextActionIdx = epsilonGreedy(Q, nextState, epsilon)

            tdTarget = reward + gamma * Q[nextState[0], nextState[1], nextActionIdx]
            tdError = tdTarget - Q[state[0], state[1], actionIdx]
            Q[state[0], state[1], actionIdx] += alpha * tdError

            state, actionIdx = nextState, nextActionIdx

        valueDeltas.append(np.abs(np.max(Q, axis=2) - prevV).flatten())

    return Q, valueDeltas


def runQLearning(numEpisodes=500, alpha=0.1, epsilon=0.1):
    Q = np.zeros((gridSize, gridSize, len(actions)))
    valueDeltas = []

    for _ in range(numEpisodes):
        prevV = np.max(Q, axis=2).copy()

        state = randomStartState()

        while state not in terminationStates:
            actionIdx = epsilonGreedy(Q, state, epsilon)
            nextState, reward = step(state, actionIdx)

            bestNextValue = np.max(Q[nextState[0], nextState[1]])
            tdTarget = reward + gamma * bestNextValue
            tdError = tdTarget - Q[state[0], state[1], actionIdx]
            Q[state[0], state[1], actionIdx] += alpha * tdError

            state = nextState

        valueDeltas.append(np.abs(np.max(Q, axis=2) - prevV).flatten())

    return Q, valueDeltas


def greedyPolicyGrid(Q):
    """Grid of each state's greedy action index, for printing."""
    return np.argmax(Q, axis=2)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    sarsaQ, sarsaDeltas = runSARSA()
    qlearnQ, qlearnDeltas = runQLearning()

    print("SARSA state values V(s) = max_a Q(s,a):")
    print(np.round(np.max(sarsaQ, axis=2), 2))
    print("\nSARSA greedy policy (0=up, 1=down, 2=right, 3=left):")
    print(greedyPolicyGrid(sarsaQ))

    print("\nQ-learning state values V(s) = max_a Q(s,a):")
    print(np.round(np.max(qlearnQ, axis=2), 2))
    print("\nQ-learning greedy policy (0=up, 1=down, 2=right, 3=left):")
    print(greedyPolicyGrid(qlearnQ))

    sarsaSeries = rollingMean([d.mean() for d in sarsaDeltas])
    qlearnSeries = rollingMean([d.mean() for d in qlearnDeltas])
    plotSeries(
        {
            "SARSA (on-policy)": (sarsaSeries, SARSA_COLOR),
            "Q-learning (off-policy)": (qlearnSeries, QLEARN_COLOR),
        },
        title="Convergence: SARSA vs Q-learning",
        ylabel="Avg |ΔV(s)| (10-episode rolling mean)",
        path="td_control_convergence.png",
        xlabel="Episode",
    )

    plotValueGrids(
        [
            (np.max(sarsaQ, axis=2), greedyPolicyGrid(sarsaQ), "SARSA"),
            (np.max(qlearnQ, axis=2), greedyPolicyGrid(qlearnQ), "Q-learning"),
        ],
        terminationStates,
        path="td_control_values.png",
        suptitle="Learned state values V(s) and greedy policy",
        gridSize=gridSize,
    )
