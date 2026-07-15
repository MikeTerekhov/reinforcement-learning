import pystk
import numpy as np
import random
from .utils import PyTux
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
        random.seed()
    
    
    def getReward(self, aim_point, current_vel, kart):
        self.reward = 0
        self.reward += -1
        #if abs(self.lastAimPoint) >= abs(aim_point[0]) and self.movement.rescue == False:
         #   reward += 50
        #else:
         #   reward += -20
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
        return self.reward
    
    
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
        a = -10000000000000
        b = -10000000000000
        
        for i in self.actions:
            if self.getQValue(state, i)[0] > a:
                a = self.getQValue(state, i)[0]
                chosenAction1 = i
            if self.getQValue(state, i)[1] > b:
                b = self.getQValue(state, i)[1]
                chosenAction2 = i
            chosen_actions = [chosenAction1, chosenAction2]
            rand_idx = random.randint(0,1)

        return chosen_actions[rand_idx]
    
    
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
      if np.random.rand() < 0.5:
        # Update (A)
        self.Qvalues_A[self.states.index(state), self.actions.index(action)] =  self.getQValue(state, action)[0] + self.alpha *(reward + (self.gamma *self.Qvalues_B[self.states.index(nextState), self.actions.index(action)] - self.Qvalues_A[self.states.index(state), self.actions.index(action)]))
        return self.getQValue(nextState, action)[0]
      else:
        # Update (B)
        self.Qvalues_B[self.states.index(state), self.actions.index(action)] =  self.getQValue(state, action)[1] + self.alpha *(reward + (self.gamma *self.Qvalues_A[self.states.index(nextState), self.actions.index(action)] - self.Qvalues_B[self.states.index(state), self.actions.index(action)]))
        return self.getQValue(nextState, action)[1]
      
    
    
    def getNextState(self, state, action):
        if action[0] == "F":
            return state
        if action[0] == "S":
            if action[1] == "N":
                if (self.states.index(state) + 1) < len(self.states): 
                    return self.states[self.states.index(state) + 1]
                else:
                    return state
            else:
                if (self.states.index(state) - 1) >= 0:
                    return self.states[self.states.index(state) - 1]
                else:
                    return state
        if action[0] == "B":
            return state
    
    
    def getState(self, aim_point, aim_point2):
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
    action = ReinLearn.getAction(state)
    ReinLearn.lastAction = action
    nextState = ReinLearn.getNextState(state, action)
    ReinLearn.adjustMovement(action)
    reward = ReinLearn.getReward(aim_point, current_vel, kart)
    #if abs(aim_point[0]) > .95:
        #ReinLearn.movement.rescue = True
    if ReinLearn.RescueTimeout < 3:
        ReinLearn.update(state, action, nextState, reward)
    #print(movement)
    

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
    for i in range(0, loop_num):
      print("LOOP " + str(i + 1))
      print(ReinLearn.epsilon)
      for t in tracks:
          #t = pystk.Track
          steps, how_far = pytux.rollout(t, control, max_frames=10000, verbose=verbose)
          print(t.capitalize() + ": " + str((steps / 10)) + "s")
          np.save('QvalueA.npy', ReinLearn.Qvalues_A)
          np.save('QvalueB.npy', ReinLearn.Qvalues_B)
          #print("A: " , ReinLearn.Qvalues_A)
          #print("B: ", ReinLearn.Qvalues_B)
          
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
