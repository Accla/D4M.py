from D4M.assoc import *

# ## Mathematical Operations on Associative Arrays

# This file demos some of the mathematical operations on Associative Array.
# 
# First we read our CSV file into an associative array again.

A = readcsv("A.csv")
A.printfull()

# For the purposes of performing mathematical operations, the string values of A must be replaced with real numbers.
# The .logical method gives a new array with 1's at each entry of the sparse matrix whch contained a non-zero value,
# and 0 otherwise.

A = A.logical()
A.printfull()

# Sum down rows and across columns

print(A.sum(0))
print(A.sum(1))

# Compute a simple join

Aa = A[:, "a,"]
Ab = A[:, "b,"]
Aab = Aa.nocol() & Ab.nocol()

# Compute a histogram (facets) of other columns that are in rows with both a and b

F = Aab.transpose() * A
F.printfull()

# Compute normalized histogram

Fn = F.divide(A.sum(0))
Fn.printfull()

# Compute correlation

AtA = A.sqin()
AtA = AtA - AtA.diag()
AtA.printfull()
