"""Visual proof that DP, SARSA, and Q-learning converge to the same policy.

418_Code_1.py solves the 4x4 gridworld with a full model of the environment
(Value Iteration). td_control.py solves the identical MDP model-free, by
sampling transitions (SARSA and Q-learning). The README claims all three
land on essentially the same optimal values and policy - this script makes
that claim visible in one figure instead of asking the reader to eyeball
three separate consoles/plots.

418_Code_1.py can't be imported directly (a Python module name can't start
with a digit), so Value Iteration is re-run here with a compact ~15-line
implementation built on the same step()/gridSize/terminationStates used by
td_control.py - the SARSA/Q-learning routines themselves are imported
unchanged from there.
"""

import random
import numpy as np

from td_control import (
    gridSize, terminationStates, actions, rewardSize, gamma,
    step, runSARSA, runQLearning, greedyPolicyGrid,
)
from viz_common import plotValueGrids


def runValueIteration(numIterations=100):
    """Tabular Value Iteration on the same MDP, using the known model."""
    V = np.zeros((gridSize, gridSize))
    states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

    for _ in range(numIterations):
        newV = np.copy(V)
        for state in states:
            if state in terminationStates:
                continue
            actionValues = []
            for actionIdx in range(len(actions)):
                nextState, reward = step(state, actionIdx)
                actionValues.append(reward + gamma * V[nextState[0], nextState[1]])
            newV[state[0], state[1]] = max(actionValues)
        V = newV

    policy = np.zeros((gridSize, gridSize), dtype=int)
    for state in states:
        actionValues = [
            step(state, a)[1] + gamma * V[step(state, a)[0][0], step(state, a)[0][1]]
            for a in range(len(actions))
        ]
        policy[state[0], state[1]] = int(np.argmax(actionValues))

    return V, policy


def policyAgreement(policyA, policyB):
    """Fraction of non-terminal states where two greedy policies agree."""
    agree, total = 0, 0
    for i in range(gridSize):
        for j in range(gridSize):
            if [i, j] in terminationStates:
                continue
            total += 1
            agree += int(policyA[i, j] == policyB[i, j])
    return agree / total


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    dpV, dpPolicy = runValueIteration()
    sarsaQ, _ = runSARSA()
    qlearnQ, _ = runQLearning()

    sarsaV, sarsaPolicy = np.max(sarsaQ, axis=2), greedyPolicyGrid(sarsaQ)
    qlearnV, qlearnPolicy = np.max(qlearnQ, axis=2), greedyPolicyGrid(qlearnQ)

    print("Greedy policy agreement with Value Iteration (ground truth):")
    print(f"  SARSA:      {policyAgreement(dpPolicy, sarsaPolicy):.0%}")
    print(f"  Q-learning: {policyAgreement(dpPolicy, qlearnPolicy):.0%}")

    plotValueGrids(
        [
            (dpV, dpPolicy, "Value Iteration\n(known model)"),
            (sarsaV, sarsaPolicy, "SARSA\n(on-policy TD)"),
            (qlearnV, qlearnPolicy, "Q-learning\n(off-policy TD)"),
        ],
        terminationStates,
        path="compare_all_values.png",
        suptitle="Three algorithms, one optimal policy",
        gridSize=gridSize,
    )
