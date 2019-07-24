from D4M.assoc import *

# ## Graph Test
# 
# Show different ways to index an associative array

# Read CSV file into associative array

E = readcsv('Edge.csv')
E.printfull()

# Get vertices and convert to numbers.

Ev = E[:, startswith('V,')].logical()

# Compute vertex adjacency graph

Av = Ev.sqin()
Av.printfull()

# Compute edge adjacency graph

Ae = Ev.sqout()
Ae.printfull()

# Get orange edges

Eo = E[(E[:,'Color,'] == 'Orange').row, :]
Eo.printfull()

# Get orange and green edges

Eog = E[startswith('O,G,'),:]
Eog.printfull()

# Another way to get orange and green edges.

EvO = Ev[startswith('O,'),:]
EvG = Ev[startswith('G,'),:]

AvOG = EvO.transpose() * EvG
print(AvOG)

AeOG = EvO * EvG.transpose()
AeOG.printfull()

# Compute edge adjacency graph preserving pedigree.

AeOG = EvO.catkeymul(EvG.transpose())
AeOG.printfull()

