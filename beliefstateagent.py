# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions, GhostRules
import numpy as np
from pacman_module import util


class BeliefStateAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        """
            Variables to use in 'updateAndFetBeliefStates' method.
            Initialization occurs in 'get_action' method.
        """
        # Current list of belief states over ghost positions
        self.beliefGhostStates = None
        # Grid of walls (assigned with 'state.getWalls()' method) 
        self.walls = None

    def updateAndGetBeliefStates(self, evidences):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of (noised) ghost positions at state x_{t}
          where 't' is the current time step

        Return:
        -------
        - A list of Z belief states at state x_{t} about ghost positions
          as N*M numpy matrices of probabilities
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """
        # XXX: Your code here
        width = self.walls.width
        height = self.walls.height
        w = 1
        p = 0.5
        pastBeliefStates = self.beliefGhostStates


        beliefStates = list()
        for i in range(len(evidences)):
            prob = np.zeros((width, height))
            pastProb = pastBeliefStates[i]
            evidence = evidences[i]

            k = 0
            for x in range(evidence[0] - w, evidence[0] + w):
                for y in range(evidence[1] - w, evidence[1] + w):
                    if evidence[0]//width == x//width and evidence[1]//height == y//height:
                        prob[x][y] = 1
                        k += 1

            for x in range(width):
                for y in range(height):
                    if prob[x][y] != 0:
                        prob[x][y] *= self.forwarding(x, y, p, pastProb)

            alpha = 1/np.sum(prob)
            # Normalization of the probability of the evidence
            for x in range(width):
                for y in range(height):
                    if prob[x][y] != 0:
                        prob[x][y] /= alpha
            print(np.sum(prob))
            beliefStates.append(prob)

        # XXX: End of your code
        self.beliefGhostStates = beliefStates
        return beliefStates

    def forwarding(self, x, y, p, pastProb):

        (width, height) = pastProb.shape

        pEast = 0
        nbAdj = 0  # Nb of valid adjacent tiles

        if (x-1)//width == x//width:
            nbAdj += 1
            pEast = p 
        if (x+1)//width == x//width:
            nbAdj += 1
        if (y-1)//height == y//height:
            nbAdj +=1
        if (y+1)//height == y//height:
            nbAdj +=1

        summation = 0

        if (x-1)//width == x//width:
            summation += pastProb[x-1][y] * ( pEast + (1-pEast)/nbAdj )
        if (x+1)//width == x//width:
            summation += pastProb[x+1][y] * (1-pEast)/nbAdj
        if (y-1)//height == y//height:
            summation += pastProb[x][y-1] * (1-pEast)/nbAdj
        if (y+1)//height == y//height:
            summation += pastProb[x][y+1] * (1-pEast)/nbAdj

        return summation

    def _computeNoisyPositions(self, state):
        """
            Compute a noisy position from true ghosts positions.
            XXX: DO NOT MODIFY THAT FUNCTION !!!
            Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        w = self.args.w

        div = float(w * w)
        new_positions = []
        for p in positions:
            (x, y) = p
            dist = util.Counter()
            for i in range(x - w, x + w):
                for j in range(y - w, y + w):
                    dist[(i, j)] = 1.0 / div
            new_positions.append(util.chooseFromDistribution(dist))
        return new_positions

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        """
           XXX: DO NOT MODIFY THAT FUNCTION !!!
                Doing so will result in a 0 grade.
        """

        # XXX : You shouldn't care on what is going on below.
        # Variables are specified in constructor.
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()
        return self.updateAndGetBeliefStates(
            self._computeNoisyPositions(state))
