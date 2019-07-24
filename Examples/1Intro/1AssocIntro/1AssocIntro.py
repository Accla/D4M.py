from D4M.assoc import *
import scipy.sparse as sp

# ## Creating Associative Arrays

# Hello! This is a test on the basic Assoc Array construction.
# Associative Array takes on entries of triplets, and it will parse an array of substrings
# that is divided by char divider.
# --Please note that this divider is indicated as the last char in the string.

row = "a,a,a,a,a,a,a,aa,aaa,b,bb,bbb,a,aa,aaa,b,bb,bbb,"
column = "a,aa,aaa,b,bb,bbb,a,a,a,a,a,a,a,aa,aaa,b,bb,bbb,"
values = "a-a,a-aa,a-aaa,a-b,a-bb,a-bbb,a-a,aa-a,aaa-a,b-a,bb-a,bbb-a,a-a,aa-aa,aaa-aaa,b-b,bb-bb,bbb-bbb,"

# Create assoc array and list triples
A = Assoc(row, column, values)

# This is the data structure of the Associative Array Class

print(A)

# The Assoc.printfull() method allows it to be printed in a tabular form.

A.printfull()

# When written into CSV form, the data is stored in the tabular form

writecsv(A, "A.csv")

# ## Subreferencing of Associative Arrays

# Subarrays of Associative Arrays have much of the same syntax as with matrices.
# 
# We start by reading our CSV file into an associative array.

A = readcsv("A.csv")
A.printfull()

# Get rows a and b

A1r = A["a,b,", :]
A1r.printfull()

# Get rows containing a and columns 1 through 3. (Note, this *includes* column 3.)

A2r = A[contains("a,"), 1:3]
A2r.printfull()

# Get rows a to b

A3r = A["a,:,b,", :]
A3r.printfull()

# Get rows starting with a or c

A4r = A[startswith("a,c,"), :]
A4r.printfull()

# Get cols a and b

A1c = A[:, "a,b,"]
A1c.printfull()

# Get rows 1 through 3 and cols containing a.

A2c = A[1:3, contains("a,")]
A2c.printfull()

# Get cols ab to c.

A3c = A[:, "ab,:,c,"]
A3c.printfull()

# Get cols starting with a or b.

A4c = A[:, startswith("a,b,")]
A4c.printfull()

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
d = AtA.adj.diagonal()
AtA = AtA.setadj(AtA.adj - sp.diags(d, shape=AtA.adj.shape))
AtA.printfull()
