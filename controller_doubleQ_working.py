import pystk
import numpy as np
import random
import matplotlib.pyplot as plt
from .utils import PyTux

# ---- chart styling, matching the gridworld scripts elsewhere in this repo ----
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"
REWARD_COLOR = "#2a78d6"   # blue
EPSILON_COLOR = "#c26a1d"  # amber


def plotTrainingProgress(rewardHistory, epsilonHistory, path="training_progress.png"):
    """Reward-per-loop and epsilon decay over training, on twin y-axes."""
    fig, axLeft = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    axLeft.set_facecolor(SURFACE)
    axRight = axLeft.twinx()

    axLeft.plot(rewardHistory, color=REWARD_COLOR, linewidth=2, solid_capstyle="round",
                label="Total reward per loop")
    axRight.plot(epsilonHistory, color=EPSILON_COLOR, linewidth=2, solid_capstyle="round",
                 label="Epsilon (exploration rate)")

    axLeft.set_title("Double Q-learning training progress", color=INK_PRIMARY, fontsize=13,
                      loc="left", pad=12)
    axLeft.set_xlabel("Outer loop", color=INK_SECONDARY, fontsize=10)
    axLeft.set_ylabel("Total reward", color=REWARD_COLOR, fontsize=10)
    axRight.set_ylabel("Epsilon", color=EPSILON_COLOR, fontsize=10)

    axLeft.grid(True, color=GRID_COLOR, linewidth=1, linestyle="-")
    axLeft.set_axisbelow(True)
    for ax in (axLeft, axRight):
        ax.spines["top"].set_visible(False)
    axLeft.spines["left"].set_color(REWARD_COLOR)
    axLeft.spines["right"].set_visible(False)
    axLeft.spines["bottom"].set_color(GRID_COLOR)
    axRight.spines["right"].set_color(EPSILON_COLOR)
    axLeft.tick_params(axis="y", colors=REWARD_COLOR, labelsize=9)
    axLeft.tick_params(axis="x", colors=INK_MUTED, labelsize=9)
    axRight.tick_params(axis="y", colors=EPSILON_COLOR, labelsize=9)

    linesLeft, labelsLeft = axLeft.get_legend_handles_labels()
    linesRight, labelsRight = axRight.get_legend_handles_labels()
    legend = axLeft.legend(linesLeft + linesRight, labelsLeft + labelsRight,
                            frameon=False, loc="lower right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


class QLearner():
    def __init__ (self):
        self.tracklength = 0
        self.epsilon = .99
        self.alpha = .1
        self.gamma = .75
        self.actionTimeout = 0
        self.lastAction = 'F50'
        self.lastAimPoint = 0
        self.actions = ["F25", "F50", "F75", "F100", "SN10", "SN5", "SP5", "SP10", "SP10D", "SN10D"]
        self.states = ["L4G", "L3G", "L2G", "L1G", "F1G", "L4L", "L3L", "L2L", "L1L", "F1L", "R1G", "R2G", "R3G", "R4G", "R1L", "R2L", "R3L", "R4L"]
        self.Qvalues_A = np.zeros((len(self.states), len(self.actions)))
        self.Qvalues_B = np.zeros((len(self.states), len(self.actions)))
        #np.save('Qvalue', self.Qvalues)
        self.movement = pystk.Action()
        self.distanceLastTraveled = 0
        self.RescueTimeout = 0
        self.totalReward = 0
        # bookkeeping for the delayed update: the effect of an action is only
        # observable on the NEXT frame, so we remember (state, action) until then
        self.prevState = None
        self.prevAction = None
        self.lastDistance = None
        random.seed()


    def startEpisode(self):
        """Forget the previous frame so the first frame of a new rollout is not
        treated as the continuation of the last one."""
        self.prevState = None
        self.prevAction = None
        self.lastDistance = None


    def getReward(self, kart):
        """Reward for the action taken on the previous frame.

        The old reward was a constant -1 per frame, which is identical for every
        action, so the Q-tables could never prefer one action over another. Here
        the reward is the distance gained along the track since the last frame
        (negative when the kart goes backwards or is rescued back up the track),
        minus a small per-frame time penalty.
        """
        distance = getattr(kart, 'overall_distance', None)
        if distance is None:
            distance = kart.distance_down_track
        progress = 0.0 if self.lastDistance is None else distance - self.lastDistance
        self.lastDistance = distance

        self.reward = progress - 0.1
        self.totalReward += self.reward
        return self.reward


    def handleRescue(self, aim_point):
        if self.movement.rescue == True:
            self.actionTimeout = 0
            #self.movement.rescue = False
            self.RescueTimeout += 1
            if self.RescueTimeout > 13:
                self.movement.rescue = True
                self.RescueTimeout = 0
            else:
                self.movement.rescue = False
            self.movement.acceleration = 1
        #if current_vel > 13:
        #    reward += 15
        self.actionTimeout += 1
        self.lastAimPoint = aim_point[0]
    
    
    def getQValue(self, state, action):
        return [self.Qvalues_A[self.states.index(state), self.actions.index(action)], self.Qvalues_B[self.states.index(state), self.actions.index(action)]]
    
    
    def computeValueFromQValues(self, state):
        a = -1000
        b = -1000
        for i in self.actions:
            if self.getQValue(state, i)[0] > a:
                a = self.getQValue(state, i)[0]
            if self.getQValue(state, i)[1] > b:
                b = self.getQValue(state, i)[1]
        a_b = [a, b]
        rand_idx = random.randint(0,1)
        
        return a_b[rand_idx]


    def computeActionFromQValues(self, state):
        # Double Q-learning acts greedily on the combined estimate Q_A + Q_B
        s = self.states.index(state)
        combined = self.Qvalues_A[s] + self.Qvalues_B[s]
        return self.actions[int(np.argmax(combined))]
    
    
    def getAction(self, state):
        if self.actionTimeout >= 15 or True:
            #print(self.lastAction)
            self.actionTimeout = 0
            if random.random() <= self.epsilon:
              return random.choice(self.actions)
            else:
              return self.computeActionFromQValues(state)
            
        else:
            self.actionTimeout += 1
            return self.lastAction
        
    def update(self, state, action, nextState, reward):
        """Double Q-learning update (van Hasselt, 2010).

        Pick one table at random to update. That table chooses the best action
        in the observed next state, and the OTHER table supplies that action's
        value. Splitting "which action is best" from "how good is it" is what
        removes the maximisation bias of ordinary Q-learning.
        """
        s = self.states.index(state)
        a = self.actions.index(action)
        ns = self.states.index(nextState)

        if np.random.rand() < 0.5:
            update_table, eval_table = self.Qvalues_A, self.Qvalues_B
        else:
            update_table, eval_table = self.Qvalues_B, self.Qvalues_A

        best_next = int(np.argmax(update_table[ns]))
        target = reward + self.gamma * eval_table[ns, best_next]
        update_table[s, a] += self.alpha * (target - update_table[s, a])
    
    
    def getState(self, aim_point, aim_point2):
        # clamp to the screen range so an aim point slightly off-screen still
        # lands in an edge bucket instead of falling through and returning None
        aim_point = [max(-1.0, min(1.0, aim_point[0]))] + list(aim_point[1:])
        aim_point2 = [max(-1.0, min(1.0, aim_point2[0]))] + list(aim_point2[1:])
        if aim_point[0] >= -1 and aim_point[0] < -.80 and aim_point[0] > aim_point2[0]:
            return self.states[0]
        elif aim_point[0] >= -.80 and aim_point[0] < -.60 and aim_point[0] > aim_point2[0]:
            return self.states[1]
        elif aim_point[0] >= -.60 and aim_point[0] < -.40 and aim_point[0] > aim_point2[0]:
            return self.states[2]
        elif aim_point[0] >= -.40 and aim_point[0] < -.20 and aim_point[0] > aim_point2[0]:
            return self.states[3]
        elif aim_point[0] >= -.20 and aim_point[0] < .20 and aim_point[0] > aim_point2[0]:
            return self.states[4]
        elif aim_point[0] >= .20 and aim_point[0] < .40 and aim_point[0] > aim_point2[0]:
            return self.states[10]
        elif aim_point[0] >= .40 and aim_point[0] <= .60 and aim_point[0] > aim_point2[0]:
            return self.states[11]
        elif aim_point[0] >= .60 and aim_point[0] <= .80 and aim_point[0] > aim_point2[0]:
            return self.states[12]
        elif aim_point[0] >= .80 and aim_point[0] <= 1 and aim_point[0] > aim_point2[0]:
            return self.states[13]
        elif aim_point[0] >= -1 and aim_point[0] < -.80 and aim_point[0] <= aim_point2[0]:
            return self.states[5]
        elif aim_point[0] >= -.80 and aim_point[0] < -.60 and aim_point[0] <= aim_point2[0]:
            return self.states[6]
        elif aim_point[0] >= -.60 and aim_point[0] < -.40 and aim_point[0] <= aim_point2[0]:
            return self.states[7]
        elif aim_point[0] >= -.40 and aim_point[0] < -.20 and aim_point[0] <= aim_point2[0]:
            return self.states[8]
        elif aim_point[0] >= -.20 and aim_point[0] <= .20 and aim_point[0] <= aim_point2[0]:
            return self.states[9]
        elif aim_point[0] >= .20 and aim_point[0] <= .40 and aim_point[0] <= aim_point2[0]:
            return self.states[14]
        elif aim_point[0] >= .40 and aim_point[0] <= .60 and aim_point[0] <= aim_point2[0]:
            return self.states[15]
        elif aim_point[0] >= .60 and aim_point[0] <= .80 and aim_point[0] <= aim_point2[0]:
            return self.states[16]
        elif aim_point[0] >= .80 and aim_point[0] <= 1 and aim_point[0] <= aim_point2[0]:
            return self.states[17]
    
    
    def adjustMovement(self, action):
        if action == "F25":
            self.movement.acceleration = .25
            self.movement.steer = 0
            self.movement.brake = False
            self.movement.drift = False
        elif action == "F50":
            self.movement.acceleration = .5
            self.movement.steer = 0
            self.movement.brake = False
            self.movement.drift = False
        elif action == "F75":
            self.movement.acceleration = .75
            self.movement.steer = 0
            self.movement.brake = False
            self.movement.drift = False
        elif action == "F100":
            self.movement.acceleration = 1
            self.movement.steer = 0
            self.movement.brake = False
            self.movement.drift = False
        elif action == "SN10":
            self.movement.acceleration = .25
            self.movement.steer = -1
            self.movement.brake = False
            self.movement.drift = False
        elif action == "SN5":
            self.movement.acceleration = .25
            self.movement.steer = -.5
            self.movement.brake = False
            self.movement.drift = False
        elif action == "SP5":
            self.movement.acceleration = .25
            self.movement.steer = .5
            self.movement.brake = False
            self.movement.drift = False
        elif action == "SP10":
            self.movement.acceleration = .25
            self.movement.steer = 1
            self.movement.brake = False
            self.movement.drift = False
        elif action == "SP10D":
            self.movement.acceleration = .25
            self.movement.steer = 1
            self.movement.brake = False
            self.movement.drift = True
        elif action == "SN10D":
            self.movement.acceleration = .25
            self.movement.steer = -1
            self.movement.brake = False
            self.movement.drift = True
        #elif action == "BT":
            #self.movement.acceleration = 0.001
            #self.movement.steer = 0
            #self.movement.brake = True
            #self.movement.drift = False



ReinLearn = QLearner()
        
def control(aim_point, aim_point2, current_vel, kart):
    movement = ReinLearn.movement
    state = ReinLearn.getState(aim_point, aim_point2)

    # The state we observe now is the real next state of last frame's action,
    # and the reward is what that action earned. Learn from that transition
    # before choosing this frame's action.
    reward = ReinLearn.getReward(kart)
    if ReinLearn.prevState is not None and ReinLearn.RescueTimeout < 3:
        ReinLearn.update(ReinLearn.prevState, ReinLearn.prevAction, state, reward)

    action = ReinLearn.getAction(state)
    ReinLearn.lastAction = action
    ReinLearn.adjustMovement(action)
    ReinLearn.handleRescue(aim_point)

    ReinLearn.prevState = state
    ReinLearn.prevAction = action
    return movement

# def test_controller(pytux, track, verbose=False):
#     import numpy as np

#     track = [track] if isinstance(track, str) else track

#     for t in track:
#         steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=verbose)
#         print(steps, how_far)

def test_controller(pytux, tracks, verbose=False):
    import numpy as np
    global run_number
    run_number = 1
    loop_num = 100
    run_number += 1
    rewardHistory = []
    epsilonHistory = []
    for i in range(0, loop_num):
      print("LOOP " + str(i + 1))
      print(ReinLearn.epsilon)
      ReinLearn.totalReward = 0
      for t in tracks:
          #t = pystk.Track
          ReinLearn.startEpisode()
          steps, how_far = pytux.rollout(t, control, max_frames=10000, verbose=verbose)
          print(t.capitalize() + ": " + str((steps / 10)) + "s")
          np.save('QvalueA.npy', ReinLearn.Qvalues_A)
          np.save('QvalueB.npy', ReinLearn.Qvalues_B)
          #print("A: " , ReinLearn.Qvalues_A)
          #print("B: ", ReinLearn.Qvalues_B)

      rewardHistory.append(ReinLearn.totalReward)
      epsilonHistory.append(ReinLearn.epsilon)
      plotTrainingProgress(rewardHistory, epsilonHistory)
      ReinLearn.epsilon *= 0.99
    
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')

    pytux = PyTux()
    test_controller(pytux, **vars(parser.parse_args()))
    pytux.close()
    
# import pystk

# from .utils import PyTux
# import math
# import numpy as np
# def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
#     """
#     Set the Action for the low-level controller
#     :param aim_point: Aim point, in screen coordinate frame [-1..1]
#     :param current_vel: Current velocity of the kart
#     :return: a pystk.Action (set acceleration, brake, steer, drift)
#     """
#     action = pystk.Action()

#     """
#     Your code here
#     Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
#     Hint: Use action.brake to True/False to brake (optionally)
#     Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
#     Hint: You may want to use action.drift=True for wide turns (it will turn faster)
#     """
#     if current_vel != 15:
#       action.acceleration = 1

#     angle = -1*(math.atan(aim_point[0]/aim_point[1]))
    
#     if angle < -0.7 or angle > 0.7:
#       action.drift = True
#     if angle < -0.90:
#       action.steer = -1
#       action.brake = True
#     elif angle > 0.90:
#       action.steer = 1
#       action.brake = True
#     else:
#       action.steer = angle
#       action.drift = False
#     # action.acceleration = np.clip(target_vel - current_vel ,0,1)
    
#     # if current_vel > target_vel:
#     #   action.brake = True
#     #   action.nitro = False
#     # else:
#     #   action.brake = False	
#     #   action.nitro = True
    
    
#     # # Compute steering
#     # action.steer = np.clip(steer_gain * aim_point[0], -1, 1)

#     # # Compute skidding
#     # if abs(aim_point[0]) > skid_thresh:
#     #     action.drift = True
  
#     # else:
#     #     action.drift = False

#     return action
  

# def test_controller(pytux, track, verbose=False):
#     import numpy as np

#     track = [track] if isinstance(track, str) else track

#     for t in track:
#         steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=verbose)
#         print(steps, how_far)

# if __name__ == '__main__':
#     from argparse import ArgumentParser

#     parser = ArgumentParser()
#     parser.add_argument('track', nargs='+')
#     parser.add_argument('-v', '--verbose', action='store_true')

#     pytux = PyTux()
#     test_controller(pytux, **vars(parser.parse_args()))
#     pytux.close()
