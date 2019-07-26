# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print ("###############################")
        #print ("successorGameState    ", successorGameState)
        #print ("newPos                ", newPos)
        #print ("newFood               ", newFood) 
        #print ("newGhostStates        ", newGhostStates)
        #print ("newScaredTimes        ", newScaredTimes)
        "*** YOUR CODE HERE ***"
        food_grid = newFood.asList()
        #for ghostState in newGhostStates:
        #    print(ghostState)
        min_food_distance = -1
        for food in food_grid:
            distance = util.manhattanDistance(food, newPos)
            if min_food_distance == -1 or distance < min_food_distance:
                min_food_distance = distance

        distance_to_ghost   = 1
        proximity_to_ghosts = 0
        for ghost_state in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost_state)
            distance_to_ghost += distance
            if distance <= 1:
                proximity_to_ghosts += 1

        return successorGameState.getScore() + (1/float(min_food_distance)) - proximity_to_ghosts - (1/float(distance_to_ghost))

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def minimax(self, gameState, agent, depth):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
           return self.evaluationFunction(gameState)
        if agent == 0:
            return max(self.minimax(gameState.generateSuccessor(0, action),1,depth) for action in gameState.getLegalActions(0))
        else:
            nextAgent = agent + 1
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0
            if nextAgent == 0:
               depth += 1
            return min(self.minimax(gameState.generateSuccessor(agent,action), nextAgent, depth) for action in gameState.getLegalActions(agent))

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        maximun = float("-inf")
        action = Directions.WEST
        for legalaction in gameState.getLegalActions(0):
            utility = self.minimax(gameState.generateSuccessor(0,legalaction),1,0)
            if utility > maximun or   maximun == float("-inf") :
                maximun = utility
                action = legalaction
        return action            
         

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_value(self,gameState, a, b, agent, depth):
        v = float("-inf")
        for action in gameState.getLegalActions(agent):
            v = max(v,self.value(gameState.generateSuccessor(0,action),a,b,1,depth))
            if v > b:
                return v
            a = max(a,v)
        return v 


    def min_value(self,gameState, a, b, agent, depth):
        v = float("inf")
        nextAgent = agent + 1
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0
            depth += 1
        for action in gameState.getLegalActions(agent):
            v = min(v,self.value(gameState.generateSuccessor(agent,action),a,b,nextAgent,depth))
            if v < a:
                return v
            b = min(b,v)
        return v

    def value(self,gameState,a,b,agent,depth):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            return self.max_value(gameState,a,b,agent,depth)
        else:
            return self.min_value(gameState,a,b,agent,depth)      

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        utility = float("-inf")
        a = float("-inf")
        b = float("inf")
        action = Directions.WEST
        for legalaction in gameState.getLegalActions(0):
            ghostValue = self.value(gameState.generateSuccessor(0,legalaction),a,b,1,0)
            if ghostValue > utility:
                utility = ghostValue      
                action = legalaction
            a = max(a,utility)
        return action  
 
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState, agent, depth):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0 :
            return max(self.expectimax(gameState.generateSuccessor(agent, action),1,depth) for action in gameState.getLegalActions(agent))
        else:
            nextAgent = agent + 1
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0
                depth += 1
            return sum(self.expectimax(gameState.generateSuccessor(agent, action), nextAgent,depth) for action in gameState.getLegalActions(agent)) / float(len(gameState.getLegalActions(agent)))  


    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        utility = float("-inf")
        action = Directions.WEST
        for legalaction in gameState.getLegalActions(0):
            value = self.expectimax(gameState.generateSuccessor(0,legalaction), 1, 0)
            if value > utility or utility == float("-inf"):
                utility = value
                action = legalaction
        return action 

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()


    newFood = currentGameState.getFood()


    food_grid = newFood.asList()

    distance_to_food = -1

    for food in food_grid:
        distance = util.manhattanDistance(newPos,food)
        if distance_to_food == -1 or distance_to_food >= distance:
            distance_to_food = distance
    
    distance_to_ghost = 1
    proximity_to_ghosts = 0
    
    for ghost_state in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(newPos, ghost_state)
        distance_to_ghost += distance
        if distance <= 1:
            proximity_to_ghosts += 1

    newCapsule = currentGameState.getCapsules()

    numberOfCapsules = len(newCapsule)
    return currentGameState.getScore() + (1 / float(distance_to_food)) - (1 / float(distance_to_ghost)) - proximity_to_ghosts - numberOfCapsules
 
# Abbreviation
better = betterEvaluationFunction
