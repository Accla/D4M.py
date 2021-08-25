from D4M.assoc import *

# Load Associative Array
E = readcsv('E.csv')

print(E.size())  # Print number of non-zero rows and columns of E
print(E.nnz())  # Print number of entries of E

col2type(E, '/').logical().sum(0)  # Count each type

En = E.logical().sum(0)  # Count each entity

tmp, entity, count = En.find()  # Get triples

An = Assoc(count, entity, 1)  # Create count by entity array
