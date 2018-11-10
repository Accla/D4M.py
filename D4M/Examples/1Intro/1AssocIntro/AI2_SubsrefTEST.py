from D4M.assoc import *

# ## Subreferencing of Associative Arrays

# Subarrays of Associative Arrays have much of the same syntax as with matrices.
# 
# We start by reading our CSV file into an associative array.

A = readcsv("A.csv")
A.printfull()

# Get rows a and b

A1r = A["a,b,",:]
A1r.printfull()

# Get rows containing a and columns 1 thru 3. (Note, this *includes* column 3.)

A2r = A[contains("a,"), 1:3]
A2r.printfull()

# Get rows a to b

A3r = A["a,:,b,",:]
A3r.printfull()

# Get rows starting with a or c

A4r = A[startswith("a,c,"),:]
A4r.printfull()

# Get cols a and b

A1c = A[:,"a,b,"]
A1c.printfull()

# Get rows 1 thru 3 and cols containing a.

A2c = A[1:3,contains("a,")]
A2c.printfull()

# Get cols ab to c.

A3c = A[:,"ab,:,c,"]
A3c.printfull()

# Get cols starting with a or b.

A4c = A[:,startswith("a,b,")]
A4c.printfull()

