# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions, GhostRules
import numpy as np
from pacman_module import util
import os  # @TODO Suppress this little piece of code
import matplotlib.pyplot as plt

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
        p = self.p
        pastBeliefStates = self.beliefGhostStates


        beliefStates = list()
        for i in range(len(evidences)):
            prob = np.zeros((width, height))
            pastProb = pastBeliefStates[i]
            evidence = evidences[i]
            for x in range(evidence[0] - w, evidence[0] + w + 1):
                for y in range(evidence[1] - w, evidence[1] + w + 1):
                    if x in range(width) and y in range(height):
                            prob[x][y] = 1

            for x in range(width):
                for y in range(height):
                    if prob[x][y] != 0:
                        prob[x][y] *= self.forwarding(x, y, p, pastProb)

            alpha = 1/np.sum(prob)
            # Normalization of the probability of the evidence
            for x in range(width):
                for y in range(height):
                    if prob[x][y] != 0:
                        prob[x][y] *= alpha
            beliefStates.append(prob)

        # XXX: End of your code
        self.beliefGhostStates = beliefStates
        return beliefStates

    def forwarding(self, x, y, p, pastProb):

        (width, height) = pastProb.shape
        sum = 0

        for i in (x - 1, x + 1):
            if i in range(width):
                sum += self.computeTerm(i, y, x, y, p, pastProb)

        for j in (y - 1, y + 1):
            if j in range(height):
                sum += self.computeTerm(x, j, x, y, p, pastProb)

        return sum

    def computeTerm(self, i, j, x, y, p, pastProb):

        nbAdj = 0
        (width, height) = pastProb.shape

        if (i+1) in range(width):
            nbAdj += 1
        if (i-1) in range(width):
            nbAdj += 1
        if (j-1) in range(height):
            nbAdj += 1
        if (j+1) in range(height):
            nbAdj += 1

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

        if self.i < 25:
            debug = ret[0]
            self.l.append(np.max(debug))
            self.i += 1
            #if debug == 1: # To Stop as soon as convergence happens
                #self.i = 25

        prefix = 'data/'  # To indicate path

        if self.i == 25:

            if os.path.exists(os.path.join(prefix, str(self.w) + "-" + str(self.p) + ".txt")):
                os.remove(os.path.join(prefix, str(self.w) + "-" + str(self.p) + ".txt"))

            f = open(os.path.join(prefix, str(self.w) + "-" + str(self.p) + ".txt"), "a")
            first = True
            for data in self.l:
                if first:
                    first = False
                    f.write(str(data))
                else:
                    f.write("," + str(data))
            self.i += 1
            f.close()
            print("Done")
            plt.plot(range(1, len(self.l)+1), self.l)
            plt.xlabel('Time step')
            plt.ylabel('Maximum probability')
            plt.title('Bayes Filter')
            plt.axis([0, self.i, 0, 1])
            plt.savefig(os.path.join(prefix, str(self.w) + "-" + str(self.p) + ".pdf"), bbox_inches='tight')
            plt.show()

        return ret
