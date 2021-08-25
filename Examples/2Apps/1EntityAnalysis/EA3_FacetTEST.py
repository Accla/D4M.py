from D4M.assoc import *

# Entity facet search.
# Shows next most common terms.

E = readcsv('E.csv')
E = E.logical()

x = 'LOCATION/new york,'
p = 'PERSON/michael chang,'
F = (E[:, x].nocol() & E[:, p].nocol()).transpose() * E  # Compute Facets
(F.transpose() > 1).printfull()  # Display most common

Fn = F.divide(E.sum(0))  # Normalize by entity counts
(Fn.transpose() > 0.02).printfull()  # Display most significant
