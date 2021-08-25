from D4M.assoc import *
from D4M.util import startswith

# Compute graphs from entity edge data
E = readcsv('E.csv')
E = E.logical()

# Compute entity (all facet pairs)
A = E.sqin()
A = A - A.diag()  # Subtract diagonal and put back in assoc.

# Normalized Correlation
d = A.adj.diagonal()
i, j, v = A.adj.find()  # Get triples from sparse matrix
An = A.setadj(sparse.coo_matrix((np.multiply(v, np.reciprocal(np.min(d[i], d[j]))), (i, j))))  # Normalize values

# Multi-facet queries
x = 'LOCATION/new york,'  # Pick location
p = startswith('PERSON/,')  # Limit to People
((A[p, x] > 4) & (An[p, x] > 0.3)).printfull()  # Find high correlations

# Triangles
p0 = 'PERSON/john kennedy,'  # Pick a person

p1 = (A[p, p0]+A[p0, p]).row  # Get neighbors of x
A[p1, p1].spy()  # Plot neighborhood

p2 = (A[p1, p1] - (A[p, p0] + A[p0, p])).row  # Get triangles of x
(A[p2, p2] > 1).printfull()  # Show popular triangles
