# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent

import random
import util
import math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - getQValue
        - getAction
        - getValue
        - getPolicy
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.gamma (discount rate)

      Functions you should use
        - self.getLegalActions(state) 
          which returns legal actions
          for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.epsilon = args['epsilon']
        self.alpha = args['alpha']
        self.gamma = args['gamma']
        self.actionFn = args['actionFn']

        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)    
          Should return 0.0 if we never seen
          a state or (state,action) tuple 
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if (state, action) in self.q_values:
            return self.q_values[(state, action)]
        else:
            return 0.0

    def getValue(self, state):
        """
          Returns max_action Q(state,action)        
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state):
            return 0.0
        return max([self.getQValue(state, action) for action in self.getLegalActions(state)])
        #legalActions = self.getLegalActions(state)
        # if len(legalActions) == 0:
        #    return 0.0
        #tmp = util.Counter()
        # for action in legalActions:
        #    tmp[action] = self.getQValue(state, action)
        # return tmp[tmp.argMax()]
        # util.raiseNotDefined()

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        bestValues = self.getValue(state)
        bestActions = list()
        for action in legalActions:
            if self.getQValue(state, action) == bestValues:
                bestActions.append(action)
        return random.choice(bestActions)

        #actions = self.getLegalActions(state)
        #best_action = None
        #max_val = float('-inf')
        # for action in actions:
        #    q_value = self.q_values[(state, action)]
        #    if max_val < q_value:
        #        max_val = q_value
        #        best_action = action
        # return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        # util.raiseNotDefined()
        legalActions = self.getLegalActions(state)
        action = None
        if not legalActions:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.getPolicy(state)
        #legal_actions = self.getLegalActions(state)
        #explore = util.flipCoin(self.epsilon)
        # if explore:
        #    return random.choice(legal_actions)
        # else:
        #    return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a 
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        # util.raiseNotDefined()
        Qpart1 = (1.0 - self.alpha) * self.getQValue(state, action)
        maxQnext = self.getQValue(nextState, self.getPolicy(nextState))
        Qpart2 = self.alpha * (reward + self.gamma * maxQnext)
        self.q_values[(state, action)] = Qpart1 + Qpart2
        #old_q_value = self.getQValue(state, action)
        #old_part = (1 - self.alpha) * old_q_value
        #reward_part = self.alpha * reward
        # if not nextState:
        #    self.q_values[(state, action)] = old_part + reward_part
        # else:
        #    nextState_part = self.alpha * \
        #        self.gamma * self.getValue(nextState)
        #    self.q_values[(state, action)] = old_part + \
        #        reward_part + nextState_part
