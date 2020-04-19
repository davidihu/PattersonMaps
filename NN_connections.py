# author: David Hurwitz
# started: 8/15/18
#

import csv
import numpy as np
from numpy import float32

# this class has the connections between all atoms in a molecule
class ProteinConnections:

    #---------------------------------------------------------------------------
    # initialize a molecule's connections. the file passed to this constructor
    # has the full set of connections for a single protein.
    #
    # after the constructor runs, ProteinConnections.m_connections[]
    # will look like this:
    # [ [1], [0,2,3], [1,5], ... ]
    #---------------------------------------------------------------------------
    def __init__(self, file):

        # list of lists.  each row is a list of a single atom's connections
        self.m_connections = []

        # read the connections of each atom from file
        with open(file, 'r') as atomConnections:
            reader = csv.reader(atomConnections)
            next(reader)  # skip the header

            # for each row of the connections file
            for row in reader:

                # get the connections for a single atom
                alist = []
                for i in range(4):
                    val = int(row[i+1])
                    if (val > -1):
                        alist.append(val)
                    else:
                        break

                # append this list of connections to the full list
                self.m_connections.append(alist)

    #---------------------------------------------------------------------------
    # return the list of connections for a single atom
    #---------------------------------------------------------------------------
    def getConnections(self, atomIndex):
        assert(atomIndex < len(self.m_connections))
        return(self.m_connections[atomIndex])

    #---------------------------------------------------------------------------
    # return the number of atoms with connections
    #---------------------------------------------------------------------------
    def getNumAtomsWithConnections(self):
        return(len(self.m_connections))

    #---------------------------------------------------------------------------
    # get the total number of atom connections.
    # this double-counts because, e.g., 1-2 is counted as is 2-1
    #---------------------------------------------------------------------------
    def getTotalNumConnections(self):
        total = 0
        for i in range(self.getNumAtomsWithConnections()):
            total = total + len(self.getConnections(i))
        return(total)

    #---------------------------------------------------------------------------
    # check if atomIndex1 and atomIndex2 are connected
    # return: True or False
    #---------------------------------------------------------------------------
    def areConnected(self, atomIndex1, atomIndex2):
        oneAtomsConnections = self.m_connections[atomIndex1]
        for i in range(len(oneAtomsConnections)):
            if oneAtomsConnections[i] == atomIndex2:
                return(True)
        return(False)
