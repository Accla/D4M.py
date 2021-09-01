from D4M.assoc import *
from D4M.util import startswith

# Compute graphs from entity edge data

E = readcsv('E.csv')
Es = E.copy()
E = E.logical()

# Entity-entity graph
Ae = E.sqin() 
Ae.printfull()
Ae.spy()

# Entity-entity graph with pedigree
p = startswith('PERSON|j,')  # Set entity range
Ep = E[:, p]  # Limit to entity range
Ap = Ep.transpose().catkeymul(Ep)  # Correlated while preserving pedigree
Ap.spy()

# Entity-entity graphs that preserve original values
Esp = Es[:, p]  # limit to entity range
Asp = Esp.transpose().catvalmul(Esp)  # Preserve original value.
Asp.spy()

# Document-document graphs
Ad = Ep.sqout()
Ad.spy()
