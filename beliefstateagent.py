# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions, GhostRules
import numpy as np
from pacman_module import util
import os  # @TODO Suppress this little piece of code
import matplotlib.pyplot as plt
import math

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
        # Uniform distribution size parameter 'w'
        # for sensor noise (see instructions)
        self.w = self.args.w
        # Probability for 'leftturn' ghost to take 'EAST' action
        # when 'EAST' is legal (see instructions)
        self.p = self.args.p
        self.i = 0  # @TODO erase it
        self.l = list()
        self.v = list()

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

        beliefStates = self.beliefGhostStates
        # XXX: Your code here
        width = self.walls.width
        height = self.walls.height
        w = self.w
        pastBeliefStates = beliefStates
        beliefStates = list()

        for i in range(len(evidences)):
            prob = np.zeros((width, height))
            pastProb = pastBeliefStates[i]
            evidence = evidences[i]
            for x in range(evidence[0] - w, evidence[0] + w + 1):
                for y in range(evidence[1] - w, evidence[1] + w + 1):
                    if x in range(width) and y in range(height):
                        prob[x][y] = 1

            backup = np.copy(prob)

            # Result of the filter without normalization
            for x in range(width):
                for y in range(height):
                    if prob[x][y] != 0:
                        prob[x][y] *= self.forwarding(x, y, pastProb)

            # If all the probabilities in the matrix are null
            if np.sum(prob) == 0:
                prob = backup

            # Normalization of the probability matrix
            alpha = 1/np.sum(prob)
            for x in range(width):
                for y in range(height):
                    if prob[x][y] != 0:
                        prob[x][y] *= alpha
            beliefStates.append(prob)

        # XXX: End of your code
        self.beliefGhostStates = beliefStates
        return beliefStates

    def forwarding(self, x, y, pastProb):
        """
        Given a position coordinates and a previous result of
        the Bayes filter, returns the sum part of the filter formula
        corresponding to the given coordinate
        Arguments:
        ----------
        - `x`: index of the column of the maze corresponding
        to the evidence  at time step t
        - `y`: index of the row of the maze corresponding
        to the evidence at time step t
        - `pastProba`: Belief state matrix at time step t-1
        Return:
        -------
        - A double representing the sum part of the filter formula
        """

        (width, height) = pastProb.shape
        sum = 0

        for i in (x - 1, x + 1):
            if i in range(width):
                sum += self.computeTerm(i, y, x, y, pastProb)

        for j in (y - 1, y + 1):
            if j in range(height):
                sum += self.computeTerm(x, j, x, y, pastProb)

        return sum

    def computeTerm(self, i, j, x, y, pastProb):
        """
        Given two position coordinates and a previous result of
        the Bayes filter, return a probability
        Arguments:
        ----------
        - `i`: index of the column of the maze corresponding
        to the evidence  at time step t-1
        - `j`: index of the row of the maze corresponding
        to the evidence at time step t-1
        - `x`: index of the column of the maze corresponding
        to the evidence  at time step t
        - `y`: index of the row of the maze corresponding
        to the evidence at time step t
        - `pastProba`: Belief state matrix at time step t-1
        Return:
        -------
        - A float corresponding to the probability of going from
        (i,j) position at previous time step to (x,y) position
        according to the transition model times the previous
        result of the Bayes filter for the position (x,y)
        """

        nbAdj = 0
        (width, height) = pastProb.shape
        p = self.p

        # Probability null for going in a wall from a valid cell
        if self.walls[x][y]:
            return 0

        # Counting the number of valid neighbour
        if (i+1) in range(width) and not self.walls[i+1][j]:
            nbAdj += 1
        if (i-1) in range(width) and not self.walls[i-1][j]:
            nbAdj += 1
        if (j-1) in range(height) and not self.walls[i][j-1]:
            nbAdj += 1
        if (j+1) in range(height) and not self.walls[i][j+1]:
            nbAdj += 1

        # According to the transition model
        if x == i+1 and y == j:
            prob = (p + (1-p)/nbAdj) * pastProb[i][j]
        else:
            prob = (1-p)/nbAdj * pastProb[i][j]

        return prob

    def _computeNoisyPositions(self, state):
        """
            Compute a noisy position from true ghosts positions.
            XXX: DO NOT MODIFY THAT FUNCTION !!!
            Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        w = self.args.w
        w2 = 2*w+1
        div = float(w2 * w2)
        new_positions = []
        for p in positions:
            (x, y) = p
            dist = util.Counter()
            for i in range(x - w, x + w + 1):
                for j in range(y - w, y + w + 1):
                    dist[(i, j)] = 1.0 / div
            dist.normalize()
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

            # @TODO Put this back to normal
        ret = self.updateAndGetBeliefStates(
            self._computeNoisyPositions(state))

        limit = 250

        if self.i < limit:
            debug = ret[0].copy()
            sum = 0
            for column in debug:
                for elem in column:
                    if elem != 0:
                        sum += elem * math.log2(elem)
            sum = -sum
            self.l.append(sum)
            # buff = list()
            # #self.l.append(np.max(debug))
            #
            # for k in range(len(debug)):
            #     for l in range(len(debug[0])):
            #         if debug[k][l] != 0:
            #             buff.append((debug[k][l], k*len(debug[0]) + l))
            #
            # mean = 0
            # var = 0
            # for k in range(len(buff)):
            #     mean += buff[k][0] * buff[k][1]
            #
            # for k in range(len(buff)):
            #     var += buff[k][0] * (buff[k][1] - mean)**2
            #
            # self.l.append(mean)
            # self.v.append(var)
            self.i += 1
            #if debug == 1: # To Stop as soon as convergence happens
                #self.i = 25

        prefix = 'data/'  # To indicate path

        if self.i == limit:

            # if os.path.exists(os.path.join(prefix, "mv" + str(self.w) + "-" + str(self.p) + ".txt")):
            #     os.remove(os.path.join(prefix, "mv" + str(self.w) + "-" + str(self.p) + ".txt"))
            #
            # f = open(os.path.join(prefix, "mv" + str(self.w) + "-" + str(self.p) + ".txt"), "a")
            # first = True
            # for data in self.l:
            #     if first:
            #         first = False
            #         f.write(str(data))
            #     else:
            #         f.write("," + str(data))
            #f.close()
            self.i += 1
            print("Done")
            plt.plot(range(1, len(self.l)+1), self.l, 'b')
            plt.xlabel('Time step')
            plt.ylabel('Entropy')
            plt.title('Bayes Filter: Entropy')
            #plt.axis([0, self.i, 0, self.wal0ls.width * self.walls.height - 1])
            plt.axis([0, limit, 0, 5])
            plt.savefig(os.path.join(prefix, "en" + str(self.w) + "-" + str(int(self.p*100)) + ".pdf"), bbox_inches='tight')
            plt.show()

        return ret
