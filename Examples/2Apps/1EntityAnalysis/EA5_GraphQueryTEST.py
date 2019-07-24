import scipy.sparse as sparse
import numpy as np
from D4M.assoc import *

# Compute graphs from entity edge data
E = readcsv('E.csv')
E = E.logical()

# Compute entity (all facet pairs)
A = E.sqin()
d = diag(A.adj) # Get diagonal of sparse matrix
A = A.setadj(A.adj - sparse.diag(d,shape=A.adj.shape)) # Subtract diagonal and put back in assoc.

# Normalized Correlation
i, j, v = A.adj.find() # Get triples from sparse matrix
An = A.setadj(sparse.coo_matrix((np.multiply(v,np.reciprocal(np.min(d[i],d[j]))), (i,j)))) # Normalize values

# Multi-facet queries
x = 'LOCATION/new york,' # Pick location
p = startswith('PERSON/,') # Limit to People
((A[p,x] > 4) & (An[p,x] > 0.3)).printfull() # Find high correlations

# Triangles
p0 = 'PERSON/john kennedy,' # Pick a person

p1 = (A[p,p0]+A[p0,p]).row # Get neighbors of x
A[p1,p1].spy() # Plot neighborhood

p2 = (A[p1,p1] - (A[p,p0] + A[p0,p])).row # Get triangles of x
(A[p2,p2] > 1).printfull() # Show popular triangles

