# author: David Hurwitz
# started: 8/15/18
#

from NN_connections import ProteinConnections

File = "C:/Users/david/Documents/newff/results/NN/simulations/mol006_sim000.connections.csv"

# get all the protein's connections
Connections = ProteinConnections(File)

# can get atom connections with ProteinConnections class member function
list0 = Connections.getConnections(0)
list1 = Connections.getConnections(1)
list2 = Connections.getConnections(2)
list3 = Connections.getConnections(3)
list152 = Connections.getConnections(152)
list153 = Connections.getConnections(153)

# or can get atom connections directly from ProteinConnections class member variable
test0 = Connections.m_connections[0]
test3 = Connections.m_connections[3]

# how many atoms have connections?
num = Connections.getNumAtomsWithConnections()

# what is the total number of atom-connections
total = Connections.getTotalNumConnections()

test = 1
