# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        i = 0
        while i < self.iterations:
            newValues = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                actions = self.mdp.getPossibleActions(state)
                bestValue = max([self.getQValue(state,action) for action in actions])
                newValues[state] = bestValue
            self.values = newValues
            i += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #print("*******************************")
        #print(action,state)
        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state,action):
            #print (nextState,prob) 
            reward = self.mdp.getReward(state,action,nextState)
            qValue += prob*(reward + self.discount*self.values[nextState])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        policies = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            policies[action] = self.getQValue(state,action)
        return policies.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        i = 0
        listState = self.mdp.getStates()
        
        while i < self.iterations:
            #print (i)
            cur_state = listState[i%len(listState)]
            if self.mdp.isTerminal(cur_state):
                i += 1
                continue
            actions = self.mdp.getPossibleActions(cur_state)
            bestValue = max([self.getQValue(cur_state,action) for action in actions])
            self.values[cur_state] = bestValue
            i += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        #print(AsynchronousValueIterationAgent)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #print("***********************")
        #print(self.mdp.getStates())
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = []
            
           
            for action in self.mdp.getPossibleActions(state):
                #print (state,action)
                nextState = state
                
                    
                if action == 'east':
                   nextState = (nextState[0],nextState[1]+1)
                elif action == 'west':
                   nextState = (nextState[0],nextState[1]-1)
                elif action == 'north':
                   nextState = (nextState[0]+1,nextState[1])
                elif action == 'south':
                   nextState = (nextState[0]-1,nextState[1])
                else :
                   continue
                if nextState in self.mdp.getStates():  
                   for nState, prob in  self.mdp.getTransitionStatesAndProbs(state,action):
                       print ("nState, prob ",state, nState, prob)
                   predecessors[state].append(nextState)
           
        #print(predecessors)
                       


        q = util.PriorityQueue()
        for state in self.mdp.getStates():
            #print (state)
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            #print(actions)
            diff = abs(max([self.getQValue(state,action) for action in actions])-self.values[state])
            q.update(state, -diff)
        #print("*************************")
        for i in range(self.iterations):
            if q.isEmpty():
                break
            state = q.pop()
            #print(state)
            actions = self.mdp.getPossibleActions(state)
            #print("actions          ", actions)
            bestValue = max([self.getQValue(state,action) for action in actions])
            self.values[state] = bestValue
            print("predecessors[state]      ", predecessors[state], "     ", state)
            for nextState in predecessors[state]:
                #print("+++++++++++++++++++++++",nextState) 
                newActions = self.mdp.getPossibleActions(nextState)
                #print("newActions", newActions)
                diff = abs(max([self.getQValue(nextState,action) for action in newActions])-self.values[nextState])
                if diff > self.theta:
                    q.update(nextState,-diff)
