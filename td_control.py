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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- chart styling (validated categorical + sequential palette) ----
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"
SARSA_COLOR = "#2a78d6"      # categorical slot 1 (blue)
QLEARN_COLOR = "#008300"     # categorical slot 2 (green)
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
ARROWS = {0: "↑", 1: "↓", 2: "→", 3: "←"}  # up, down, right, left

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


def rollingMean(values, window=10):
    """Smooth a noisy per-episode series so the trend is readable."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plotConvergence(sarsaDeltas, qlearnDeltas, path="td_control_convergence.png"):
    """Smoothed per-episode |delta V(s)|, SARSA vs Q-learning, one axis, two series."""
    sarsaSeries = rollingMean([d.mean() for d in sarsaDeltas])
    qlearnSeries = rollingMean([d.mean() for d in qlearnDeltas])

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(sarsaSeries, color=SARSA_COLOR, linewidth=2, solid_capstyle="round",
            label="SARSA (on-policy)")
    ax.plot(qlearnSeries, color=QLEARN_COLOR, linewidth=2, solid_capstyle="round",
            label="Q-learning (off-policy)")

    ax.set_title("Convergence: SARSA vs Q-learning", color=INK_PRIMARY, fontsize=13,
                 loc="left", pad=12)
    ax.set_xlabel("Episode", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Avg |ΔV(s)| (10-episode rolling mean)", color=INK_SECONDARY, fontsize=10)

    ax.grid(True, color=GRID_COLOR, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    legend = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plotValueGrids(sarsaQ, qlearnQ, path="td_control_values.png"):
    """Side-by-side heatmaps of the learned V(s) with the greedy policy overlaid."""
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    worstV = min(np.max(sarsaQ, axis=2).min(), np.max(qlearnQ, axis=2).min())

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), facecolor=SURFACE)

    for ax, Q, name in zip(axes, [sarsaQ, qlearnQ], ["SARSA", "Q-learning"]):
        V = np.max(Q, axis=2)
        policy = np.argmax(Q, axis=2)
        magnitude = -V  # 0 (best, near a goal) -> light ... worst -> dark
        ax.imshow(magnitude, cmap=cmap, vmin=0, vmax=-worstV)

        for i in range(gridSize):
            for j in range(gridSize):
                normVal = magnitude[i, j] / -worstV if worstV != 0 else 0
                textColor = "#ffffff" if normVal > 0.55 else INK_PRIMARY
                if [i, j] in terminationStates:
                    ax.text(j, i, "GOAL", ha="center", va="center", color=INK_PRIMARY,
                            fontsize=9, fontweight="bold")
                else:
                    ax.text(j, i, f"{ARROWS[policy[i, j]]}\n{V[i, j]:.1f}", ha="center",
                            va="center", color=textColor, fontsize=10)

        ax.set_title(name, color=INK_PRIMARY, fontsize=12, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Learned state values V(s) and greedy policy", color=INK_PRIMARY,
                 fontsize=13, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


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

    plotConvergence(sarsaDeltas, qlearnDeltas)
    plotValueGrids(sarsaQ, qlearnQ)
